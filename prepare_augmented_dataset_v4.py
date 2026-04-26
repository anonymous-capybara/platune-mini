#!/usr/bin/env python
"""
Chained multi-augmentation pipeline: applies up to six augmentations
sequentially to each waveform. Each enabled augmentation is always
applied, with strength controlled by a Beta(a, 1) distribution.

Augmentation presets are loaded from a gin config under
`platune/configs/augmentations/`. Runtime and I/O options remain on the CLI.

Augmentation order (fixed):
  1. White noise   (additive Gaussian noise)
  2. Filter        (pedalboard LowpassFilter / HighpassFilter, 50/50)
  3. Pitch shift   (pedalboard.PitchShift)
  4. Chorus        (pedalboard.Chorus)
  5. Reverb        (pedalboard.Reverb)
  6. Random mix    (weighted blend with a random waveform from the LMDB)

Each augmented entry stores an N-element alpha vector (N = number of
enabled augmentations), in the same order as AUGMENTATION_NAMES.
For example, with all six enabled:
  [alpha_noise, alpha_filter, alpha_pitch, alpha_chorus, alpha_reverb, alpha_mix]
With only [noise, pitch_shift, reverb] enabled:
  [alpha_noise, alpha_pitch, alpha_reverb]

When pitch shift is applied, the following attributes are shifted
accordingly (no BasicPitch recomputation):
  melody, melody_processed, pitch, pitch_processed, octave, octave_processed

All other attributes are copied verbatim from the source entry.

Prerequisites:
  - Source LMDB must have been prepared with --save_waveform

Usage:
    python prepare_augmented_dataset_v4.py \\
        --lmdb_path processed_data \\
        --output_path processed_data_aug_v4 \\
        --n_augmented 5000 \\
        --config augmentations_v4 \\
        --codec music2latent \\
        --gpu 0
"""

import os
import argparse
import copy
import pickle
import gin
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from platune.datasets.audio_example import AudioExample


torch.set_grad_enabled(False)

# Augmentation names (in application order) and their alpha indices
AUGMENTATION_NAMES = ["noise", "filter", "pitch_shift", "chorus", "reverb", "mix"]
N_AUGMENTATIONS = len(AUGMENTATION_NAMES)

# Standard MIDI constants for pitch/octave recomputation
N_PITCH_CLASSES = 12  # C, C#, D, ..., B
OCTAVE_BOUNDARIES = torch.tensor([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120])

# Attribute keys affected by pitch shift
PITCH_SHIFT_ATTR_KEYS = {
    "melody", "melody_processed",
    "pitch", "pitch_processed",
    "octave", "octave_processed",
}

PRIMARY_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "platune", "configs", "augmentations")


def load_codec(codec_name: str, device: str):
    if codec_name == "music2latent":
        from music2latent import EncoderDecoder
        return EncoderDecoder(device=device)

    if codec_name == "codicodec":
        from codicodec import EncoderDecoder
        return EncoderDecoder(device=device)

    raise ValueError(f"Unsupported codec: {codec_name}")


# ----------------------------------------------------------------
# Pitch-shift attribute adjustment
# ----------------------------------------------------------------
def shift_pitch_attributes(attrs: dict, semitones_shift: int) -> dict:
    """Shift melody/pitch/octave attributes by an integer number of semitones.

    Args:
        attrs: dict of attr_name -> np.ndarray (will be modified in-place)
        semitones_shift: integer semitone shift (positive = up, negative = down)

    Returns:
        attrs: the modified dict
    """
    if semitones_shift == 0:
        return attrs

    # Shift melody (raw): preserve zeros (silence)
    if "melody" in attrs:
        melody = attrs["melody"].copy()
        nonzero = melody > 0
        melody[nonzero] = np.clip(melody[nonzero] + semitones_shift, 1, 127)
        attrs["melody"] = melody.astype(np.int32)

        # Recompute pitch and octave from shifted melody
        attrs["pitch"] = (melody % N_PITCH_CLASSES).astype(np.int32)
        attrs["octave"] = (torch.bucketize(
            torch.from_numpy(melody.astype(np.int64)),
            OCTAVE_BOUNDARIES,
        ) - 1).numpy().astype(np.int32)

    # Shift melody_processed (gap-filled version): no zeros to preserve
    if "melody_processed" in attrs:
        melody_proc = attrs["melody_processed"].copy()
        melody_proc = np.clip(melody_proc + semitones_shift, 0, 127)
        attrs["melody_processed"] = melody_proc.astype(np.int32)

        # Recompute pitch_processed and octave_processed
        attrs["pitch_processed"] = (melody_proc % N_PITCH_CLASSES).astype(np.int32)
        attrs["octave_processed"] = (torch.bucketize(
            torch.from_numpy(melody_proc.astype(np.int64)),
            OCTAVE_BOUNDARIES,
        ) - 1).numpy().astype(np.int32)

    return attrs


def load_augmentation_config(config_name: str) -> dict:
    """Load augmentation settings from a gin config file."""
    config_basename = os.path.splitext(os.path.basename(config_name))[0]
    primary_config_path = os.path.join(PRIMARY_CONFIG_DIR, f"{config_basename}.gin")

    if os.path.exists(primary_config_path):
        config_path = primary_config_path
    else:
        raise FileNotFoundError(
            f"Augmentation config not found: {primary_config_path} ")

    gin.clear_config()
    gin.parse_config_files_and_bindings([config_path], [])

    def q(name: str):
        return gin.query_parameter(f"%{name}")

    enabled_augmentations = q("ENABLED_AUGMENTATIONS")
    unknown_augmentations = sorted(
        set(enabled_augmentations) - set(AUGMENTATION_NAMES))
    if unknown_augmentations:
        raise ValueError(
            f"Unknown augmentations in {config_path}: {unknown_augmentations}. "
            f"Expected subset of {AUGMENTATION_NAMES}.")
    if len(enabled_augmentations) == 0:
        raise ValueError(f"{config_path} must enable at least one augmentation.")

    return {
        "config_path": config_path,
        "enabled_augmentations": enabled_augmentations,
        "noise_min_snr_db": q("NOISE_MIN_SNR_DB"),
        "noise_max_snr_db": q("NOISE_MAX_SNR_DB"),
        "noise_beta_a": q("NOISE_BETA_A"),
        "lpf_min_freq": q("LPF_MIN_FREQ"),
        "lpf_max_freq": q("LPF_MAX_FREQ"),
        "hpf_min_freq": q("HPF_MIN_FREQ"),
        "hpf_max_freq": q("HPF_MAX_FREQ"),
        "filter_beta_a": q("FILTER_BETA_A"),
        "pitch_shift_min_semitones": q("PITCH_SHIFT_MIN_SEMITONES"),
        "pitch_shift_max_semitones": q("PITCH_SHIFT_MAX_SEMITONES"),
        "pitch_shift_beta_a": q("PITCH_SHIFT_BETA_A"),
        "chorus_rate_hz_min": q("CHORUS_RATE_HZ_MIN"),
        "chorus_rate_hz_max": q("CHORUS_RATE_HZ_MAX"),
        "chorus_depth_min": q("CHORUS_DEPTH_MIN"),
        "chorus_depth_max": q("CHORUS_DEPTH_MAX"),
        "chorus_centre_delay_min": q("CHORUS_CENTRE_DELAY_MIN"),
        "chorus_centre_delay_max": q("CHORUS_CENTRE_DELAY_MAX"),
        "chorus_feedback_min": q("CHORUS_FEEDBACK_MIN"),
        "chorus_feedback_max": q("CHORUS_FEEDBACK_MAX"),
        "chorus_mix_min": q("CHORUS_MIX_MIN"),
        "chorus_mix_max": q("CHORUS_MIX_MAX"),
        "chorus_beta_a": q("CHORUS_BETA_A"),
        "reverb_room_size_min": q("REVERB_ROOM_SIZE_MIN"),
        "reverb_room_size_max": q("REVERB_ROOM_SIZE_MAX"),
        "reverb_damping_min": q("REVERB_DAMPING_MIN"),
        "reverb_damping_max": q("REVERB_DAMPING_MAX"),
        "reverb_wet_min": q("REVERB_WET_MIN"),
        "reverb_wet_max": q("REVERB_WET_MAX"),
        "reverb_beta_a": q("REVERB_BETA_A"),
        "mix_max_alpha": q("MIX_MAX_ALPHA"),
        "mix_beta_a": q("MIX_BETA_A"),
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Chained multi-augmentation pipeline: "
                    "noise → filter → pitch_shift → chorus → reverb → mix")

    # I/O
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Path to the source LMDB directory (must contain waveforms)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for the augmented LMDB")
    parser.add_argument("--n_augmented", type=int, required=True,
                        help="Total number of augmented entries to generate")
    parser.add_argument("--config", type=str, default="augmentations_v4",
                        help="Basename of the augmentation gin config in platune/configs/augmentations")

    # Codec
    parser.add_argument("--codec", type=str, default="music2latent",
                        choices=["music2latent", "codicodec"],
                        help="Which codec to use for encoding")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU to use (-1 for CPU)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for codec encoding")
    parser.add_argument("--db_size", type=int, default=40,
                        help="Max LMDB size in GB")

    # Audio params
    parser.add_argument("--num_signal", type=int, default=131_072,
                        help="Waveform chunk length in samples")
    parser.add_argument("--sr", type=int, default=44_100,
                        help="Sample rate")

    # Anchor pool selection
    parser.add_argument("--lmdb_keys_file", type=str, default=None,
                        help="Optional pickle with anchor keys to use (basename without .pkl)")

    # Misc
    parser.add_argument("--save_waveform", action="store_true",
                        help="Also store augmented waveform in output LMDB")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")

    args = parser.parse_args()
    aug_cfg = load_augmentation_config(args.config)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "mps" if torch.mps.is_available() and args.gpu >= 0 else "cpu"
    rng = np.random.default_rng(args.seed)
    enabled = set(aug_cfg["enabled_augmentations"])

    # Build ordered list of enabled augmentations (preserving AUGMENTATION_NAMES order)
    enabled_ordered = [name for name in AUGMENTATION_NAMES if name in enabled]
    n_alpha = len(enabled_ordered)
    # Map augmentation name -> index in alpha vector
    alpha_idx = {name: i for i, name in enumerate(enabled_ordered)}

    print(f"Loading augmentation config: {aug_cfg['config_path']}")
    print(f"Enabled augmentations: {enabled_ordered}")
    print(f"Alpha vector size: {n_alpha}")

    # ----------------------------------------------------------------
    # 1. Open source LMDB and get keys
    # ----------------------------------------------------------------
    src_env = lmdb.open(args.lmdb_path, lock=False, readonly=True, readahead=False)
    with src_env.begin() as txn:
        all_keys = list(txn.cursor().iternext(values=False))
    print(f"Source LMDB: {len(all_keys)} entries")

    if args.lmdb_keys_file is not None:
        keys_path = os.path.join(args.lmdb_path, f"{args.lmdb_keys_file}.pkl")
        with open(keys_path, "rb") as f:
            all_keys = pickle.load(f)
        print(f"Loaded {len(all_keys)} keys from {keys_path}")

    anchor_keys = all_keys
    if len(anchor_keys) == 0:
        raise ValueError("No anchor keys available for augmentation.")

    # ----------------------------------------------------------------
    # 2. Verify waveforms exist
    # ----------------------------------------------------------------
    with src_env.begin() as txn:
        _test_ae = AudioExample(txn.get(anchor_keys[0]))
        try:
            _test_w = _test_ae.get("waveform")
            assert _test_w is not None
        except Exception:
            raise ValueError(
                "Source LMDB does not contain waveforms. "
                "Re-run scripts/prepare_dataset.py with --save_waveform.")

    # Discover attribute keys stored in the source LMDB
    attr_keys = [k for k in _test_ae.get_keys()
                 if k not in ("z", "waveform", "metadata", "midi")]
    print(f"Attribute keys found in source LMDB: {attr_keys}")

    # ----------------------------------------------------------------
    # 3. Load codec
    # ----------------------------------------------------------------
    print(f"Loading codec ({args.codec}) on {device}...")
    codec = load_codec(args.codec, device)
    print(f"{args.codec} loaded.")

    # Probe codec to determine z_length
    if args.codec == "codicodec":
        _probe_w = torch.zeros(1, 1, args.num_signal).to(device)
    else:
        _probe_w = torch.zeros(1, args.num_signal).to(device)
    _probe_z = codec.encode(_probe_w) if args.codec != "codicodec" else codec.encode(_probe_w, fix_batch_size=True)
    if args.codec == "codicodec":
        _probe_z = _probe_z.reshape(
            _probe_z.shape[0],
            _probe_z.shape[1] * _probe_z.shape[2],
            _probe_z.shape[3]).transpose(1, 2)
    z_length = _probe_z.shape[-1]
    del _probe_w, _probe_z
    print(f"  z_length = {z_length}")

    # ----------------------------------------------------------------
    # 4. Build augmentation objects
    # ----------------------------------------------------------------

    # 4a. White noise (additive Gaussian)
    if "noise" in enabled:
        print(
            f"  Noise: SNR [{aug_cfg['noise_min_snr_db']}, "
            f"{aug_cfg['noise_max_snr_db']}] dB")

    # 4b. Filter (pedalboard) — instantiated per-example with random cutoff
    if "filter" in enabled:
        from pedalboard import LowpassFilter, HighpassFilter
        print(
            f"  Filter: LPF [{aug_cfg['lpf_min_freq']}, "
            f"{aug_cfg['lpf_max_freq']}] Hz, "
            f"HPF [{aug_cfg['hpf_min_freq']}, {aug_cfg['hpf_max_freq']}] Hz")

    # 4c. Pitch shift (pedalboard) — instantiated per-example with random semitones
    pitch_shift_max_range = max(
        abs(aug_cfg["pitch_shift_min_semitones"]),
        abs(aug_cfg["pitch_shift_max_semitones"]))
    if "pitch_shift" in enabled:
        from pedalboard import PitchShift
        print(
            f"  PitchShift: [{aug_cfg['pitch_shift_min_semitones']}, "
            f"{aug_cfg['pitch_shift_max_semitones']}] semitones")

    # 4d. Chorus (pedalboard) — instantiated per-example with random params
    if "chorus" in enabled:
        from pedalboard import Chorus
        print(
            f"  Chorus: rate_hz [{aug_cfg['chorus_rate_hz_min']}, "
            f"{aug_cfg['chorus_rate_hz_max']}], depth "
            f"[{aug_cfg['chorus_depth_min']}, {aug_cfg['chorus_depth_max']}], "
            f"mix [{aug_cfg['chorus_mix_min']}, {aug_cfg['chorus_mix_max']}]")

    # 4e. Reverb (pedalboard) — instantiated per-example with random params
    if "reverb" in enabled:
        from pedalboard import Reverb
        print(
            f"  Reverb: wet [{aug_cfg['reverb_wet_min']}, "
            f"{aug_cfg['reverb_wet_max']}]")

    # 4f. Random mix — weighted blend with a random waveform
    if "mix" in enabled:
        print(
            f"  Mix: max_alpha={aug_cfg['mix_max_alpha']}, "
            f"beta_a={aug_cfg['mix_beta_a']}, selection=random")

    # ----------------------------------------------------------------
    # 5. Prepare output LMDB
    # ----------------------------------------------------------------
    os.makedirs(args.output_path, exist_ok=True)
    dst_env = lmdb.open(
        args.output_path,
        map_size=args.db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    # ----------------------------------------------------------------
    # 6. Generate augmented entries
    # ----------------------------------------------------------------
    print(f"\nGenerating {args.n_augmented} augmented entries...")

    # Pre-sample anchor indices
    anchor_indices = rng.integers(0, len(anchor_keys), size=args.n_augmented)

    cur_out_idx = 0
    aug_counts = {name: 0 for name in AUGMENTATION_NAMES}

    # Buffers for batched codec encoding
    batch_waveforms = []      # augmented waveforms (float32 np)
    batch_alphas = []         # N-element alpha arrays (N = len(enabled_ordered))
    batch_attrs = []          # attribute dicts
    batch_metadata = []       # metadata dicts
    batch_source_keys = []    # source LMDB keys
    batch_aug_details = []    # per-entry augmentation detail dicts

    def _flush_batch():
        """Encode batch of waveforms through codec and write to LMDB."""
        nonlocal cur_out_idx
        if len(batch_waveforms) == 0:
            return

        # Encode through codec
        w_batch_np = np.stack(batch_waveforms)  # [B, num_signal]
        w_batch_torch = torch.from_numpy(w_batch_np).float().to(device)
        if args.codec == "codicodec":
            w_batch_torch = w_batch_torch.unsqueeze(1)  # [B, 1, num_signal]
        encode_kwargs = {}
        if args.codec == "codicodec":
            encode_kwargs["fix_batch_size"] = True
        z_batch = codec.encode(w_batch_torch, **encode_kwargs)
        if args.codec == "codicodec":
            B_enc = z_batch.shape[0]
            z_batch = z_batch.reshape(
                B_enc, z_batch.shape[1] * z_batch.shape[2], 64
            ).transpose(1, 2)

        # Write each entry
        for j in range(len(batch_waveforms)):
            out_key = f"{cur_out_idx:08d}"
            ae_out = AudioExample()

            # Store z
            ae_out.put_array("z", z_batch[j].cpu().numpy(), dtype=np.float32)

            # Store N-element alpha vector (N = number of enabled augmentations)
            ae_out.put_array("alpha",
                             np.array(batch_alphas[j], dtype=np.float32),
                             dtype=np.float32)

            # Store attributes
            for ak, av in batch_attrs[j].items():
                if isinstance(av, np.ndarray):
                    if np.issubdtype(av.dtype, np.integer):
                        ae_out.put_array(ak, av, dtype=np.int32)
                    else:
                        ae_out.put_array(ak, av, dtype=np.float32)

            # Metadata
            meta_out = copy.deepcopy(batch_metadata[j])
            meta_out["augmented"] = True
            meta_out["augmentation_type"] = "chained"
            source_key = batch_source_keys[j]
            meta_out["source_key"] = (source_key.decode()
                                      if isinstance(source_key, bytes)
                                      else source_key)
            meta_out["augmentation_details"] = batch_aug_details[j]
            meta_out["enabled_augmentations"] = enabled_ordered
            meta_out["key"] = out_key
            ae_out.put_metadata(meta_out)

            # Optionally store waveform
            if args.save_waveform:
                w_int16 = (batch_waveforms[j] * (2**15 - 1)).astype(np.int16)
                ae_out.put_array("waveform", w_int16, dtype=np.int16)

            with dst_env.begin(write=True) as txn:
                txn.put(out_key.encode(), bytes(ae_out))
            cur_out_idx += 1

        batch_waveforms.clear()
        batch_alphas.clear()
        batch_attrs.clear()
        batch_metadata.clear()
        batch_source_keys.clear()
        batch_aug_details.clear()

    for i in tqdm(range(args.n_augmented), desc="Augmenting"):
        anchor_key = anchor_keys[anchor_indices[i]]

        # Load source entry
        with src_env.begin() as txn:
            ae = AudioExample(txn.get(anchor_key))
            waveform = ae.get("waveform")  # float32 [num_signal]
            try:
                metadata = ae.get_metadata()
            except Exception:
                metadata = {}

            # Load all source attributes
            source_attrs = {}
            for ak in attr_keys:
                try:
                    source_attrs[ak] = ae.get(ak).copy()
                except Exception:
                    pass

        # Initialize alpha vector (one slot per enabled augmentation)
        alphas = [0.0] * n_alpha
        aug_details = {}

        # Current waveform starts as the source
        w_current = waveform.copy()

        # --- Augmentation 1: White Noise (additive Gaussian) ---
        if "noise" in enabled:
            t = rng.beta(aug_cfg["noise_beta_a"], 1.0)  # t in [0, 1]
            snr_in_db = aug_cfg["noise_min_snr_db"] + t * (
                aug_cfg["noise_max_snr_db"] - aug_cfg["noise_min_snr_db"])
            signal_rms = np.sqrt(np.mean(w_current ** 2)) + 1e-8
            noise_rms = signal_rms * 10 ** (-snr_in_db / 20)
            w_current = w_current + rng.standard_normal(w_current.shape).astype(np.float32) * noise_rms

            alphas[alpha_idx["noise"]] = (
                aug_cfg["noise_max_snr_db"] - snr_in_db
            ) / (
                aug_cfg["noise_max_snr_db"] - aug_cfg["noise_min_snr_db"])
            aug_details["noise"] = {"snr_in_db": float(snr_in_db)}
            aug_counts["noise"] += 1

        # --- Augmentation 2: Filter (pedalboard) ---
        if "filter" in enabled:
            use_lowpass = rng.random() < 0.5

            if use_lowpass:
                t = rng.beta(aug_cfg["filter_beta_a"], 1.0)
                cutoff_hz = float(np.exp(
                    np.log(aug_cfg["lpf_min_freq"]) + t * (
                        np.log(aug_cfg["lpf_max_freq"]) -
                        np.log(aug_cfg["lpf_min_freq"]))))
                filt = LowpassFilter(cutoff_frequency_hz=cutoff_hz)
                alpha_filter = np.log(
                    aug_cfg["lpf_max_freq"] / cutoff_hz) / np.log(
                        aug_cfg["lpf_max_freq"] / aug_cfg["lpf_min_freq"])
                filter_type = "low_pass"
            else:
                t = rng.beta(aug_cfg["filter_beta_a"], 1.0)
                cutoff_hz = float(np.exp(
                    np.log(aug_cfg["hpf_min_freq"]) + t * (
                        np.log(aug_cfg["hpf_max_freq"]) -
                        np.log(aug_cfg["hpf_min_freq"]))))
                filt = HighpassFilter(cutoff_frequency_hz=cutoff_hz)
                alpha_filter = np.log(
                    cutoff_hz / aug_cfg["hpf_min_freq"]) / np.log(
                        aug_cfg["hpf_max_freq"] / aug_cfg["hpf_min_freq"])
                filter_type = "high_pass"

            w_current = filt(w_current[np.newaxis, :], args.sr).squeeze()
            alphas[alpha_idx["filter"]] = float(np.clip(alpha_filter, 0.0, 1.0))
            aug_details["filter"] = {"type": filter_type, "cutoff_hz": float(cutoff_hz)}
            aug_counts["filter"] += 1

        # --- Augmentation 3: Pitch Shift (pedalboard) ---
        if "pitch_shift" in enabled:
            t = rng.beta(aug_cfg["pitch_shift_beta_a"], 1.0)  # magnitude in [0, 1]
            if rng.random() < 0.5:
                semitones = t * aug_cfg["pitch_shift_max_semitones"]   # shift up
            else:
                semitones = -t * abs(aug_cfg["pitch_shift_min_semitones"])  # shift down
            ps = PitchShift(semitones=semitones)
            w_current = ps(w_current[np.newaxis, :], args.sr).squeeze()

            alphas[alpha_idx["pitch_shift"]] = min(abs(semitones) / pitch_shift_max_range, 1.0)

            # Shift melody/pitch/octave attributes
            semitones_int = int(round(semitones))
            source_attrs = shift_pitch_attributes(source_attrs, semitones_int)

            aug_details["pitch_shift"] = {
                "semitones": float(semitones),
                "semitones_int": semitones_int,
            }
            aug_counts["pitch_shift"] += 1

        # --- Augmentation 4: Chorus (pedalboard) ---
        if "chorus" in enabled:
            rate_hz = rng.uniform(
                aug_cfg["chorus_rate_hz_min"], aug_cfg["chorus_rate_hz_max"])
            depth = rng.uniform(
                aug_cfg["chorus_depth_min"], aug_cfg["chorus_depth_max"])
            centre_delay_ms = rng.uniform(
                aug_cfg["chorus_centre_delay_min"],
                aug_cfg["chorus_centre_delay_max"])
            feedback = rng.uniform(
                aug_cfg["chorus_feedback_min"], aug_cfg["chorus_feedback_max"])
            mix = aug_cfg["chorus_mix_min"] + rng.beta(
                aug_cfg["chorus_beta_a"], 1.0) * (
                    aug_cfg["chorus_mix_max"] - aug_cfg["chorus_mix_min"])

            chorus = Chorus(
                rate_hz=rate_hz,
                depth=depth,
                centre_delay_ms=centre_delay_ms,
                feedback=feedback,
                mix=mix,
            )
            w_current = chorus(w_current[np.newaxis, :], args.sr).squeeze()

            alphas[alpha_idx["chorus"]] = float(
                (mix - aug_cfg["chorus_mix_min"]) / (
                    aug_cfg["chorus_mix_max"] - aug_cfg["chorus_mix_min"]))
            aug_details["chorus"] = {
                "rate_hz": float(rate_hz),
                "depth": float(depth),
                "centre_delay_ms": float(centre_delay_ms),
                "feedback": float(feedback),
                "mix": float(mix),
            }
            aug_counts["chorus"] += 1

        # --- Augmentation 5: Reverb (pedalboard) ---
        if "reverb" in enabled:
            room_size = rng.uniform(
                aug_cfg["reverb_room_size_min"], aug_cfg["reverb_room_size_max"])
            damping = rng.uniform(
                aug_cfg["reverb_damping_min"], aug_cfg["reverb_damping_max"])
            wet_level = aug_cfg["reverb_wet_min"] + rng.beta(
                aug_cfg["reverb_beta_a"], 1.0) * (
                    aug_cfg["reverb_wet_max"] - aug_cfg["reverb_wet_min"])
            dry_level = 1.0

            reverb = Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
                width=1.0,
                freeze_mode=0.0,
            )
            w_current = reverb(w_current[np.newaxis, :], args.sr).squeeze()

            alphas[alpha_idx["reverb"]] = float(
                (wet_level - aug_cfg["reverb_wet_min"]) / (
                    aug_cfg["reverb_wet_max"] - aug_cfg["reverb_wet_min"]))
            aug_details["reverb"] = {
                "room_size": float(room_size),
                "damping": float(damping),
                "wet_level": float(wet_level),
                "dry_level": float(dry_level),
            }
            aug_counts["reverb"] += 1

        # --- Augmentation 6: Random Mix (weighted blend) ---
        if "mix" in enabled:
            t = rng.beta(aug_cfg["mix_beta_a"], 1.0)
            alpha_mix = t * aug_cfg["mix_max_alpha"]  # scale to [0, mix_max_alpha]

            # Pick a random waveform from the anchor pool.
            mix_key = anchor_keys[rng.integers(0, len(anchor_keys))]
            with src_env.begin() as txn:
                ae_mix = AudioExample(txn.get(mix_key))
                w_mix = ae_mix.get("waveform")  # float32 [num_signal]

            # Handle length mismatch: truncate or zero-pad
            if len(w_mix) > len(w_current):
                w_mix = w_mix[:len(w_current)]
            elif len(w_mix) < len(w_current):
                w_mix = np.pad(w_mix, (0, len(w_current) - len(w_mix)))

            w_current = (1.0 - alpha_mix) * w_current + alpha_mix * w_mix

            alphas[alpha_idx["mix"]] = float(t)  # t is already in [0, 1]
            mix_key_str = mix_key.decode() if isinstance(mix_key, bytes) else mix_key
            aug_details["mix"] = {"alpha_mix": float(alpha_mix), "mix_source_key": mix_key_str}
            aug_counts["mix"] += 1

        # Accumulate for batched codec encoding
        batch_waveforms.append(w_current)
        batch_alphas.append(alphas)
        batch_attrs.append(source_attrs)
        batch_metadata.append(metadata)
        batch_source_keys.append(anchor_key)
        batch_aug_details.append(aug_details)

        if len(batch_waveforms) >= args.batch_size:
            _flush_batch()

    # Flush remaining
    _flush_batch()

    src_env.close()
    dst_env.close()

    # Summary
    print(f"\nDone! Wrote {cur_out_idx} augmented entries to {args.output_path}")
    print(f"  Per-augmentation counts:")
    for name in AUGMENTATION_NAMES:
        if name in enabled:
            print(f"    {name}: {aug_counts[name]} ({100 * aug_counts[name] / args.n_augmented:.1f}%)")
        else:
            print(f"    {name}: disabled")


if __name__ == "__main__":
    main()
