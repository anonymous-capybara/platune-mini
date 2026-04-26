#!/usr/bin/env python
"""
Decode latent vectors from an LMDB dataset to audio files, creating
a reference audio set for KAD/FAD evaluation.

The script selects a deterministic subset of chunks (default 2000), decodes
each z via the configured codec (music2latent or codicodec), and writes .wav
files to the output directory.

Usage:
    python scripts/create_reference_audio.py \
        --lmdb_path <lmdb_dir> \
        --output_dir <output_dir> \
        --n_samples 2000 \
        --gpu 0
"""

import os
import argparse
import gin
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

from platune.datasets.dataset import LatentsContinuousDiscreteAttritbutesDataset


def main():
    parser = argparse.ArgumentParser(
        description="Decode LMDB latents to wav files for reference audio set")
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Path to LMDB database")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write reference wav files")
    parser.add_argument("--gin_config", type=str, default=None,
                        help="Path to gin config (default: auto-detect from lmdb_path or use v1.gin)")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of chunks to decode (default: 2000)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk selection")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Decoding batch size")
    parser.add_argument("--lmdb_keys_file", type=str, default=None,
                        help="Optional pickle file with filtered LMDB keys (basename without .pkl)")
    parser.add_argument("--sample_rate", type=int, default=44100,
                        help="Output sample rate")
    parser.add_argument("--codec", type=str, default="codicodec",
                        choices=["music2latent", "codicodec"],
                        help="Codec to use for decoding (default: codicodec)")
    args = parser.parse_args()

    # Resolve device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load gin config for dataset keys
    if args.gin_config is not None:
        gin_path = args.gin_config
    else:
        # Try lmdb-level config.gin, fall back to default v1.gin
        lmdb_gin = os.path.join(args.lmdb_path, "config.gin")
        if os.path.isfile(lmdb_gin):
            gin_path = lmdb_gin
        else:
            gin_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "platune", "configs", "v1.gin",
            )
    print(f"Loading gin config: {gin_path}")
    gin.clear_config()
    gin.parse_config_files_and_bindings([gin_path], [])

    DISCRETE_KEYS = gin.query_parameter('%DISCRETE_KEYS')
    CONTINUOUS_KEYS = gin.query_parameter('%CONTINUOUS_KEYS')

    # Optional filtering params
    try:
        MAX_CHUNKS = gin.query_parameter('%MAX_CHUNKS_PER_INSTRUMENT')
    except ValueError:
        MAX_CHUNKS = None
    try:
        INSTR_FILTER = gin.query_parameter('%INSTRUMENT_FILTER')
    except ValueError:
        INSTR_FILTER = None
    try:
        FILTER_SEED = gin.query_parameter('%FILTER_SEED')
    except ValueError:
        FILTER_SEED = 42

    # Load dataset
    dataset = LatentsContinuousDiscreteAttritbutesDataset(
        path=args.lmdb_path,
        keys=["z"] + DISCRETE_KEYS + CONTINUOUS_KEYS,
        lmdb_keys_file=args.lmdb_keys_file,
        max_chunks_per_instrument=MAX_CHUNKS,
        instrument_filter=INSTR_FILTER,
        filter_seed=FILTER_SEED,
    )
    dataset_len = len(dataset.keys)
    print(f"Dataset has {dataset_len} chunks")

    # Select deterministic subset
    rng = np.random.default_rng(args.seed)
    actual_n = min(args.n_samples, dataset_len)
    eval_indices = rng.choice(dataset_len, size=actual_n, replace=False)
    print(f"Selected {actual_n} chunks for reference audio")

    # Load codec
    print(f"Loading {args.codec} codec...")
    if args.codec == "codicodec":
        from codicodec import EncoderDecoder
    else:
        from music2latent import EncoderDecoder
    codec = EncoderDecoder(device=device)

    # Probe latents_per_timestep for codicodec
    if args.codec == "codicodec":
        with torch.no_grad():
            _probe = torch.randn(1, 1, 131072, device=device)
            _probe_z = codec.encode(_probe)
            latents_per_timestep = _probe_z.shape[2]
        del _probe, _probe_z
        torch.cuda.empty_cache()
        print(f"  latents_per_timestep = {latents_per_timestep}")
    else:
        latents_per_timestep = 1
    print("Codec loaded")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Decode in batches
    file_idx = 0
    for batch_start in tqdm(
        range(0, actual_n, args.batch_size),
        desc="Decoding",
    ):
        batch_end = min(batch_start + args.batch_size, actual_n)
        batch_indices = eval_indices[batch_start:batch_end]

        # Collect latents
        zs = []
        for idx in batch_indices:
            z, _ad, _ac, _alpha = dataset[idx]
            zs.append(z)

        # Pad to common T and stack
        T_max = max(z.shape[-1] for z in zs)
        zs_padded = []
        for z in zs:
            if z.shape[-1] < T_max:
                pad = z[:, -1:].expand(-1, T_max - z.shape[-1])
                z = torch.cat([z, pad], dim=-1)
            zs_padded.append(z)

        z_batch = torch.stack(zs_padded, dim=0).to(device)  # [B, D, T]

        # Decode — reshape for codicodec if needed
        with torch.no_grad():
            if latents_per_timestep > 1:
                # [B, 64, T_flat] -> [B, T_orig, lpt, 64]
                B_dec = z_batch.shape[0]
                T_flat = z_batch.shape[2]
                T_orig = T_flat // latents_per_timestep
                z_dec = z_batch.transpose(1, 2).reshape(
                    B_dec, T_orig, latents_per_timestep, -1
                )
            else:
                z_dec = z_batch
            audio_batch = codec.decode(z_dec)
            # codicodec returns [B, 2, samples] (stereo) — take first channel
            if audio_batch.dim() == 3:
                audio_batch = audio_batch[:, 0, :]
        
        # Write individual wav files
        audio_np = audio_batch.cpu().numpy()
        for j in range(len(batch_indices)):
            out_path = os.path.join(args.output_dir, f"{file_idx:04d}.wav")
            sf.write(out_path, audio_np[j], args.sample_rate)
            file_idx += 1

    print(f"\nDone. Wrote {file_idx} wav files to {args.output_dir}/")


if __name__ == "__main__":
    main()
