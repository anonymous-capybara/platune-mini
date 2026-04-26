#!/usr/bin/env python
"""
Evaluation script for PLaTune.

Supports three evaluation tasks (each opt-in via flags):

1. Style transfer (--enable_style_transfer):
   For each pair, randomly select one of three swap modes and transfer those
   controls from the target into the source while preserving the source style
   and remaining controls.
   - melody: swaps pitch_processed + octave_processed + onsets
   - dynamics: swaps dynamics
   - instrument: swaps instrument
   Saves transferred audio and expected attributes (the controls the output
   audio should exhibit) for downstream accuracy measurement.

2. Reconstruction (--enable_reconstruction):
   Round-trip z_src -> cs (z_to_cs) -> z_rec (cs_to_z) and decode.
   Measures reconstruction MSE in latent space.

3. Conditional synthesis (--enable_cond_synth):
   Assemble controls from two chunks: melody (pitch+octave+onsets) and dynamics
   from the source, instrument from the target. Style is sampled from N(0,1).
   Synthesizes via cs_to_z and decodes. Saves the synthesized audio, composite
   control tensor, and expected melody/onsets for downstream accuracy measurement.
Style transfer flow: z_src -> cs_src (via z_to_cs) -> swap control channels -> z_out (via cs_to_z)

Usage:
    python scripts/evaluate_platune.py \
        --gin_config platune/configs/v1.gin \
        --checkpoint runs/model/checkpoints/last.ckpt \
        --lmdb_path processed_data \
        --output_dir eval_outputs_platune \
        --num_samples 500 \
        --device cuda:0 \
        --enable_style_transfer \
        --enable_reconstruction \
        --enable_cond_synth
"""

import os
import json
import argparse
import gin
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

from platune.model import PLaTune
from platune.datasets.dataset import LatentsContinuousDiscreteAttritbutesDataset


def denormalize_controls_to_per_attr(c_np, model, all_keys, classes_attr_discrete):
    """
    Convert normalized control tensor to per-attribute denormalized values.

    Args:
        c_np: Normalized control tensor [control_dim, T]
        model: PLaTune model (for channel_ranges, min_max_attr)
        all_keys: List of attribute names (discrete + continuous)
        classes_attr_discrete: List of class value lists per discrete attr
        
    Returns:
        denorm: Per-attribute denormalized values [n_attributes, T]
    """
    n_attr = len(all_keys)
    T = c_np.shape[-1]
    denorm = np.zeros((n_attr, T))
    
    n_discrete = len(classes_attr_discrete)
    
    for attr_idx, key in enumerate(all_keys):
        start, end = model.channel_ranges[key]

        # Scalar channel: denormalize from [-1,1] back to class index,
        # then map through class values for discrete attributes.
        min_val, max_val = model.min_max_attr[start]
        raw = denormalize_attr(c_np[start], min_val, max_val)
        if attr_idx < n_discrete:
            classes = classes_attr_discrete[attr_idx]
            idx = np.clip(np.round(raw).astype(int), 0, len(classes) - 1)
            denorm[attr_idx] = np.array([classes[i] for i in idx])
        else:
            denorm[attr_idx] = raw
    
    return denorm


def platune_style_transfer(model, z_src, c_src, c_tgt, nb_steps=20, control_mask=None):
    """
    Perform style transfer with PLaTune.
    
    Flow: z_src -> cs_src (z_to_cs) -> swap control with mask -> z_out (cs_to_z)
    
    Args:
        model: PLaTune model
        z_src: Source latent [B, latent_dim, T]
        c_src: Source control (normalized) [B, control_dim, T] (unused if no mask)
        c_tgt: Target control (normalized) [B, control_dim, T]
        nb_steps: ODE integration steps
        control_mask: Optional [control_dim] mask for partial transfer (1 = use target, 0 = keep source)
    
    Returns:
        z_out: Transformed latent [B, latent_dim, T]
    """
    # Extract cs from source: z -> [control, style]
    cs_rec_src = model.z_to_cs(z_src, nb_steps=nb_steps)
    
    # Split into control and style
    c_rec_src = cs_rec_src[:, :model.control_dim, :]  # [B, control_dim, T]
    style_src = cs_rec_src[:, model.control_dim:, :]   # [B, style_dim, T]
    
    # Apply partial or full control swap
    if control_mask is not None:
        # Mask shape: [control_dim] -> [1, control_dim, 1] for broadcasting
        mask = control_mask.view(1, -1, 1).to(c_tgt.device)
        c_swapped = c_rec_src * (1 - mask) + c_tgt * mask
    else:
        c_swapped = c_tgt
    
    # Reconstruct cs with swapped control
    cs_swapped = torch.cat([c_swapped, style_src], dim=1)  # [B, latent_dim, T]
    
    # Transform back to latent space
    z_out = model.cs_to_z(cs_swapped, nb_steps=nb_steps)
    
    return z_out


def denormalize_attr(value_normalized, min_val, max_val):
    """Convert normalized [-1, 1] attribute back to original range."""
    # Reverse: norm = 2 * (x - min) / (max - min) - 1
    # x = (norm + 1) / 2 * (max - min) + min
    return (value_normalized + 1) / 2 * (max_val - min_val) + min_val


def reconstruct_melody_from_pitch_octave(pitch_class, octave_index):
    """
    Reconstruct MIDI note number from pitch class and octave index.
    
    Reverses process_midi_attributes logic:
        pitch = melody % 12
        octave = bucketize(melody, [0,12,24,...,120]) - 1
    So melody = 12 * octave + pitch (for pitch > 0),
       melody = 12 * (octave + 1) when pitch == 0.
    """
    pitch = np.round(pitch_class).astype(int)
    octave = np.round(octave_index).astype(int)
    melody = np.where(pitch > 0, 12 * octave + pitch, 12 * (octave + 1))
    return melody


def main():
    parser = argparse.ArgumentParser(description="Evaluate PLaTune melody transfer")
    parser.add_argument("--gin_config", type=str, required=True, help="Path to gin config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB database")
    parser.add_argument("--output_dir", type=str, default="eval_outputs_platune", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of chunk pairs to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair selection")
    parser.add_argument("--lmdb_keys_file", type=str, default=None,
                        help="Optional pickle file with filtered LMDB keys (basename without .pkl)")
    # Task flags (at least one should be enabled)
    parser.add_argument("--enable_style_transfer", action="store_true",
                        help="Enable melody+onsets style transfer task")
    parser.add_argument("--enable_reconstruction", action="store_true",
                        help="Enable round-trip reconstruction task (z -> cs -> z)")
    parser.add_argument("--enable_cond_synth", action="store_true",
                        help="Enable conditional synthesis task (mixed controls + Gaussian style)")
    parser.add_argument("--n_alpha_channels", type=int, default=0,
                        help="Number of alpha channels the model was trained with (default: 0)")
    parser.add_argument("--n_alpha_classes", type=int, default=0,
                        help="Number of alpha classes for discrete alpha mode (default: 0)")
    args = parser.parse_args()

    # Default: if no task flag is given, enable all
    if not (args.enable_style_transfer or args.enable_reconstruction or args.enable_cond_synth):
        args.enable_style_transfer = True
        args.enable_reconstruction = True
        args.enable_cond_synth = True

    enabled_tasks = []
    if args.enable_style_transfer:
        enabled_tasks.append("style_transfer")
    if args.enable_reconstruction:
        enabled_tasks.append("reconstruction")
    if args.enable_cond_synth:
        enabled_tasks.append("cond_synth")
    print(f"Enabled tasks: {enabled_tasks}")

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "source_audio"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "target_audio"), exist_ok=True)
    if args.enable_style_transfer:
        os.makedirs(os.path.join(args.output_dir, "transferred_audio"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "transferred_expected_attr"), exist_ok=True)
    if args.enable_reconstruction:
        os.makedirs(os.path.join(args.output_dir, "reconstructed_audio"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "reconstructed_expected_attr"), exist_ok=True)
    if args.enable_cond_synth:
        os.makedirs(os.path.join(args.output_dir, "cond_synth_audio"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "cond_synth_expected_attrs"), exist_ok=True)

    # Load gin config
    gin.clear_config()
    gin.parse_config_files_and_bindings([args.gin_config], [])

    # Query parameters from gin
    DISCRETE_KEYS = gin.query_parameter('%DISCRETE_KEYS')
    CONTINUOUS_KEYS = gin.query_parameter('%CONTINUOUS_KEYS')
    CLASSES_ATTR_DISCRETE = gin.query_parameter('%CLASSES_ATTR_DISCRETE')
    MAX_CHUNKS_PER_INSTRUMENT = gin.query_parameter('%MAX_CHUNKS_PER_INSTRUMENT')
    INSTRUMENT_FILTER = gin.query_parameter('%INSTRUMENT_FILTER')
    FILTER_SEED = gin.query_parameter('%FILTER_SEED')
    NB_STEPS = gin.query_parameter('%NB_STEPS')
    ALPHA_DISCRETE = gin.query_parameter('%ALPHA_DISCRETE')
    alpha_discrete = ALPHA_DISCRETE and args.n_alpha_channels > 0
    model_n_alpha = 1 if alpha_discrete else args.n_alpha_channels

    all_keys = DISCRETE_KEYS + CONTINUOUS_KEYS
    print(f"Discrete keys: {DISCRETE_KEYS}")
    print(f"Continuous keys: {CONTINUOUS_KEYS}")
    print(f"All keys: {all_keys}")

    # Verify all required keys exist in config
    REQUIRED_KEYS = ["pitch_processed", "octave_processed", "onsets", "dynamics", "instrument"]
    for key in REQUIRED_KEYS:
        if key not in all_keys:
            raise ValueError(f"Required attribute '{key}' not found in keys: {all_keys}")

    # Create dataset
    dataset = LatentsContinuousDiscreteAttritbutesDataset(
        path=args.lmdb_path,
        keys=["z"] + DISCRETE_KEYS + CONTINUOUS_KEYS,
        lmdb_keys_file=args.lmdb_keys_file,
        max_chunks_per_instrument=MAX_CHUNKS_PER_INSTRUMENT,
        instrument_filter=INSTRUMENT_FILTER,
        filter_seed=FILTER_SEED,
        n_alpha_channels=model_n_alpha,
    )
    dataset.alpha_discrete = alpha_discrete
    num_chunks = len(dataset.keys)
    print(f"Dataset has {num_chunks} chunks")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    with gin.unlock_config():
        gin.bind_parameter("model.PLaTune.n_alpha_channels", model_n_alpha)
        gin.bind_parameter("model.PLaTune.alpha_discrete", alpha_discrete)
        gin.bind_parameter("model.PLaTune.n_alpha_classes", args.n_alpha_classes)
    model = PLaTune()
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()
    print(f"Model loaded on {args.device}")
    print(f"Model control_dim: {model.control_dim}, style_dim: {model.style_dim}")
    print(f"Model alpha channels: {model.n_alpha_channels}")
    print(f"Channel ranges: {model.channel_ranges}")

    # Define two swap modes with their control masks
    # Each mode swaps a specific group of channels (1=swap from target, 0=keep from source)
    SWAP_MODES = {
        "melody": ["pitch_processed", "octave_processed", "onsets", "dynamics"],
        "instrument": ["instrument"],
    }
    swap_masks = {}
    for mode_name, mode_keys in SWAP_MODES.items():
        mask = torch.zeros(model.control_dim)
        for key in mode_keys:
            start, end = model.channel_ranges[key]
            mask[start:end] = 1.0
        swap_masks[mode_name] = mask
    swap_mode_names = list(SWAP_MODES.keys())
    print(f"Swap modes: {swap_mode_names}")
    for mode_name, mask in swap_masks.items():
        n_ones = int(mask.sum().item())
        print(f"  {mode_name}: {n_ones} channels swapped")

    # Generate deterministic pairs (src_idx, tgt_idx)
    rng = np.random.default_rng(args.seed)
    num_samples = min(args.num_samples, num_chunks)
    
    # Sample source indices, then sample different target indices
    src_indices = rng.choice(num_chunks, size=num_samples, replace=False)
    tgt_indices = []
    for src_idx in src_indices:
        tgt_idx = rng.integers(0, num_chunks)
        while tgt_idx == src_idx:
            tgt_idx = rng.integers(0, num_chunks)
        tgt_indices.append(tgt_idx)
    
    print(f"Evaluating {num_samples} chunk pairs...")

    # Per-attribute MSE accumulators for attribute extraction accuracy
    attr_mse_sum = np.zeros(len(all_keys))
    attr_mse_count = 0

    # Reconstruction MSE accumulator (latent-space round-trip error)
    recon_mse_sum = 0.0
    recon_mse_count = 0

    # Process each pair
    with torch.no_grad():
        for i, (src_idx, tgt_idx) in enumerate(tqdm(
                zip(src_indices, tgt_indices), total=num_samples)):
            # Load source and target chunks
            # Dataset returns: z, attr_discrete, attr_continuous, alpha
            z_src, ad_src, ac_src, alpha_src = dataset[src_idx]
            z_tgt, ad_tgt, ac_tgt, alpha_tgt = dataset[tgt_idx]

            # Handle different chunk lengths: use min length
            T_src = z_src.shape[-1]
            T_tgt = z_tgt.shape[-1]
            T = min(T_src, T_tgt)
            
            z_src = z_src[:, :T].unsqueeze(0).to(args.device)
            z_tgt = z_tgt[:, :T].unsqueeze(0).to(args.device)
            ad_src = ad_src[:, :T].unsqueeze(0) if ad_src.ndim == 2 else ad_src.unsqueeze(0)
            ac_src = ac_src[:, :T].unsqueeze(0) if ac_src.ndim == 2 else ac_src.unsqueeze(0)
            ad_tgt = ad_tgt[:, :T].unsqueeze(0) if ad_tgt.ndim == 2 else ad_tgt.unsqueeze(0)
            ac_tgt = ac_tgt[:, :T].unsqueeze(0) if ac_tgt.ndim == 2 else ac_tgt.unsqueeze(0)

            # In discrete alpha mode, concatenate alpha to discrete attrs
            if model.alpha_discrete and model.n_alpha_channels > 0:
                alpha_src_t = alpha_src[:, :T].unsqueeze(0)
                alpha_tgt_t = alpha_tgt[:, :T].unsqueeze(0)
                ad_src = torch.cat([ad_src, alpha_src_t.long()], dim=1)
                ad_tgt = torch.cat([ad_tgt, alpha_tgt_t.long()], dim=1)

            # Process and normalize attributes using model's methods
            a_src = model.process_attributes(ad_src, ac_src)
            c_src = model.normalize_attr(a_src)
            a_tgt = model.process_attributes(ad_tgt, ac_tgt)
            c_tgt = model.normalize_attr(a_tgt)

            # Append alpha channels (all -1.0 = no augmentation) when model expects them
            if model.n_alpha_channels > 0 and not model.alpha_discrete:
                alpha_zero = torch.full((1, model.n_alpha_channels, T), -1.0).to(args.device)
                c_src = torch.cat([c_src, alpha_zero], dim=1)
                c_tgt = torch.cat([c_tgt, alpha_zero], dim=1)

            # Attribute extraction: z_src -> cs_rec_src, then measure MSE
            cs_rec_src = model.z_to_cs(z_src, nb_steps=NB_STEPS)
            c_rec_src = cs_rec_src[:, :model.control_dim, :]   # [B, control_dim, T]
            style_src = cs_rec_src[:, model.control_dim:, :]    # [B, style_dim, T]

            # Per-attribute MSE between extracted and ground-truth controls
            c_diff_sq = (c_rec_src - c_src.to(args.device)) ** 2  # [B, control_dim, T]
            for attr_idx, key in enumerate(all_keys):
                start, end = model.channel_ranges[key]
                attr_mse_sum[attr_idx] += c_diff_sq[:, start:end, :].mean().item()
            attr_mse_count += 1

            # ── Task 1: Style transfer ──────────────────────────────────
            if args.enable_style_transfer:
                # Randomly select swap mode for this pair (uniform)
                swap_mode = swap_mode_names[rng.integers(0, len(swap_mode_names))]
                mask = swap_masks[swap_mode].view(1, -1, 1).to(args.device)

                # Swap: extracted source controls + target controls via mask
                c_swapped = c_rec_src * (1 - mask) + c_tgt.to(args.device) * mask
                cs_swapped = torch.cat([c_swapped, style_src], dim=1)
                z_transformed = model.cs_to_z(cs_swapped, nb_steps=NB_STEPS)
                audio_transformed = model.decode_latent_to_audio(z_transformed).cpu().numpy()[0]

                # Expected controls: ground-truth source + ground-truth target via mask
                # (what the transformed audio should exhibit)
                c_expected = c_src.to(args.device) * (1 - mask) + c_tgt.to(args.device) * mask

                # Denormalize expected controls to per-attribute values
                c_expected_np = c_expected[0].cpu().numpy()  # [control_dim, T]
                c_expected_denorm = denormalize_controls_to_per_attr(
                    c_expected_np, model, all_keys, CLASSES_ATTR_DISCRETE
                )  # [n_attributes, T]

                # Reconstruct expected melody from pitch + octave attributes
                pitch_attr_idx = all_keys.index("pitch_processed")
                octave_attr_idx = all_keys.index("octave_processed")
                melody_midi = reconstruct_melody_from_pitch_octave(
                    c_expected_denorm[pitch_attr_idx], c_expected_denorm[octave_attr_idx]
                )

                # Expected onsets (denormalized, rounded to binary)
                onsets_attr_idx = all_keys.index("onsets")
                onsets = np.round(c_expected_denorm[onsets_attr_idx]).astype(int)

                # Save style transfer outputs
                sf.write(
                    os.path.join(args.output_dir, "transferred_audio", f"{i:04d}_transferred.wav"),
                    audio_transformed, model.sample_rate
                )
                np.savez(
                    os.path.join(args.output_dir, "transferred_expected_attr", f"{i:04d}_attrs.npz"),
                    swap_mode=swap_mode,                            # str: melody/dynamics/instrument
                    control_mask=swap_masks[swap_mode].numpy(),     # [control_dim] binary mask
                    c_expected=c_expected_np,                       # [control_dim, T] normalized
                    c_expected_denorm=c_expected_denorm,            # [control_dim, T] denormalized
                    melody_midi=melody_midi,                       # [T] expected MIDI note values
                    onsets=onsets,                                  # [T] expected binary onsets
                    keys=np.array(all_keys),                       # attribute names
                )

            # ── Task 2: Reconstruction ────────────────────────────────────
            if args.enable_reconstruction:
                z_reconstructed = model.cs_to_z(cs_rec_src, nb_steps=NB_STEPS)
                audio_reconstructed = model.decode_latent_to_audio(z_reconstructed).cpu().numpy()[0]

                # Accumulate latent-space reconstruction MSE
                recon_mse = ((z_reconstructed - z_src) ** 2).mean().item()
                recon_mse_sum += recon_mse
                recon_mse_count += 1

                sf.write(
                    os.path.join(args.output_dir, "reconstructed_audio", f"{i:04d}_rec.wav"),
                    audio_reconstructed, model.sample_rate
                )

                # Expected attributes = source GT (reconstruction target)
                c_rec_expected_np = c_src[0].cpu().numpy()  # [control_dim, T]
                c_rec_expected_denorm = denormalize_controls_to_per_attr(
                    c_rec_expected_np, model, all_keys, CLASSES_ATTR_DISCRETE
                )  # [n_attributes, T]

                pitch_attr_idx = all_keys.index("pitch_processed")
                octave_attr_idx = all_keys.index("octave_processed")
                onsets_attr_idx = all_keys.index("onsets")
                melody_midi_rec = reconstruct_melody_from_pitch_octave(
                    c_rec_expected_denorm[pitch_attr_idx], c_rec_expected_denorm[octave_attr_idx]
                )
                onsets_rec = np.round(c_rec_expected_denorm[onsets_attr_idx]).astype(int)

                np.savez(
                    os.path.join(args.output_dir, "reconstructed_expected_attr", f"{i:04d}_attrs.npz"),
                    c_expected=c_rec_expected_np,                   # [control_dim, T] normalized
                    c_expected_denorm=c_rec_expected_denorm,        # [control_dim, T] denormalized
                    melody_midi=melody_midi_rec,                    # [T] expected MIDI note values
                    onsets=onsets_rec,                               # [T] expected binary onsets
                    keys=np.array(all_keys),                        # attribute names
                )

            # ── Task 3: Conditional synthesis ─────────────────────────────
            if args.enable_cond_synth:
                # Assemble controls from 2 GT sources (range-aware):
                #   melody (pitch+octave+onsets) + dynamics from source
                #   instrument from target
                c_synth = torch.zeros(1, model.control_dim, T, device=args.device)
                if model.n_alpha_channels > 0:
                    a_start, a_end = model.channel_ranges["alpha"]
                    c_synth[:, a_start:a_end, :] = -1.0  # all alpha channels = no augmentation
                c_src_dev = c_src.to(args.device)
                c_tgt_dev = c_tgt.to(args.device)

                # Melody from source (pitch+octave+onsets)
                for mk in ["pitch_processed", "octave_processed", "onsets"]:
                    s, e = model.channel_ranges[mk]
                    c_synth[:, s:e, :] = c_src_dev[:, s:e, :]
                # Dynamics from source
                s, e = model.channel_ranges["dynamics"]
                c_synth[:, s:e, :] = c_src_dev[:, s:e, :]
                # Instrument from target
                s, e = model.channel_ranges["instrument"]
                c_synth[:, s:e, :] = c_tgt_dev[:, s:e, :]

                # Sample style from standard Gaussian
                style_random = torch.randn(1, model.style_dim, T, device=args.device)
                cs_synth = torch.cat([c_synth, style_random], dim=1)  # [1, latent_dim, T]

                z_synth = model.cs_to_z(cs_synth, nb_steps=NB_STEPS)
                audio_synth = model.decode_latent_to_audio(z_synth).cpu().numpy()[0]

                # Denormalize expected controls to per-attribute values
                c_synth_np = c_synth[0].cpu().numpy()  # [control_dim, T]
                c_synth_denorm = denormalize_controls_to_per_attr(
                    c_synth_np, model, all_keys, CLASSES_ATTR_DISCRETE
                )  # [n_attributes, T]

                # Expected melody and onsets from source GT
                pitch_attr_idx = all_keys.index("pitch_processed")
                octave_attr_idx = all_keys.index("octave_processed")
                onsets_attr_idx = all_keys.index("onsets")
                melody_midi_synth = reconstruct_melody_from_pitch_octave(
                    c_synth_denorm[pitch_attr_idx], c_synth_denorm[octave_attr_idx]
                )
                onsets_synth = np.round(c_synth_denorm[onsets_attr_idx]).astype(int)

                # Provenance: per-attribute source labels
                source_map = np.array(["" for _ in range(len(all_keys))], dtype=object)
                source_map[pitch_attr_idx] = "src"
                source_map[octave_attr_idx] = "src"
                source_map[onsets_attr_idx] = "src"
                source_map[all_keys.index("dynamics")] = "src"
                source_map[all_keys.index("instrument")] = "tgt"

                sf.write(
                    os.path.join(args.output_dir, "cond_synth_audio", f"{i:04d}_synth.wav"),
                    audio_synth, model.sample_rate
                )
                np.savez(
                    os.path.join(args.output_dir, "cond_synth_expected_attrs", f"{i:04d}_attrs.npz"),
                    c_expected=c_synth_np,                      # [control_dim, T] normalized
                    c_expected_denorm=c_synth_denorm,           # [control_dim, T] denormalized
                    melody_midi=melody_midi_synth,              # [T] expected MIDI note values
                    onsets=onsets_synth,                        # [T] expected binary onsets
                    source_map=source_map,                      # [control_dim] provenance labels
                    keys=np.array(all_keys),                   # attribute names
                )

            # ── Common: save source & target audio ────────────────────────
            audio_src = model.decode_latent_to_audio(z_src).cpu().numpy()[0]
            audio_tgt = model.decode_latent_to_audio(z_tgt).cpu().numpy()[0]
            sf.write(
                os.path.join(args.output_dir, "source_audio", f"{i:04d}_src.wav"),
                audio_src, model.sample_rate
            )
            sf.write(
                os.path.join(args.output_dir, "target_audio", f"{i:04d}_tgt.wav"),
                audio_tgt, model.sample_rate
            )

    # ── Post-loop metrics ─────────────────────────────────────────────
    metrics = {"num_samples": attr_mse_count}

    # Attribute extraction MSE (always computed since z_to_cs is always run)
    attr_mse_mean = attr_mse_sum / max(attr_mse_count, 1)
    metrics["attr_extraction"] = {
        "overall_mse": float(attr_mse_mean.mean()),
        "per_attribute": {key: float(attr_mse_mean[j]) for j, key in enumerate(all_keys)},
    }

    print(f"\nAttribute extraction accuracy (MSE, lower = better):")
    for key, mse_val in metrics["attr_extraction"]["per_attribute"].items():
        print(f"  {key}: {mse_val:.6f}")
    print(f"  Overall: {metrics['attr_extraction']['overall_mse']:.6f}")

    # Reconstruction MSE
    if args.enable_reconstruction and recon_mse_count > 0:
        recon_mse_mean = recon_mse_sum / recon_mse_count
        metrics["reconstruction"] = {
            "latent_mse": float(recon_mse_mean),
        }
        print(f"\nReconstruction accuracy (latent MSE, lower = better):")
        print(f"  Latent MSE: {recon_mse_mean:.6f}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Summary
    print(f"\nEvaluation complete. Outputs saved to {args.output_dir}/")
    print(f"  - {num_samples} source audio files in source_audio/")
    print(f"  - {num_samples} target audio files in target_audio/")
    if args.enable_style_transfer:
        print(f"  - {num_samples} transferred audio files in transferred_audio/")
        print(f"  - {num_samples} transferred expected attr files in transferred_expected_attr/")
    if args.enable_reconstruction:
        print(f"  - {num_samples} reconstructed audio files in reconstructed_audio/")
        print(f"  - {num_samples} reconstructed expected attr files in reconstructed_expected_attr/")
    if args.enable_cond_synth:
        print(f"  - {num_samples} cond synth audio files in cond_synth_audio/")
        print(f"  - {num_samples} cond synth expected attr files in cond_synth_expected_attrs/")


if __name__ == "__main__":
    main()
