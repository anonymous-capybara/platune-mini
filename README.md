# PLaTune-mini Workflow

This repository contains three main command-line scripts for the full PLaTune workflow:

1. `prepare_dataset.py` builds the base latent LMDB from audio files.
2. `prepare_augmented_dataset_v4.py` optionally creates an augmented LMDB from that base dataset.
3. `train.py` trains the PLaTune model from the prepared LMDBs.

The examples below follow the same flow as `train-audio-latent-diffusion-full.ipynb`.

All commands below assume your current working directory is `platune-mini/`.

## 1. Prepare the base latent dataset

`prepare_dataset.py` reads audio files, splits them into fixed-length chunks, encodes each chunk with a pretrained codec, and writes the results to an LMDB.

It can also`:
- compute audio descriptors such as `rms`, `integrated_loudness`, and `loudness1s`
- extract MIDI with BasicPitch and derive MIDI-based attributes if `--use_basic_pitch --midi_attributes` flags are on

### Example

```bash
python prepare_dataset.py \
    --output_path datasets/guitatset-test \
    --input_path "/path/to/guitarset-mono-mic-small" \
    --config simple \
    --emb_model_path music2latent \
    --parser_name simple_parser \
    --gpu 0 \
    --num_signal 131072 \
    --cut_silences \
    --save_waveform \
    --use_basic_pitch --midi_attributes \
    -l rms -l integrated_loudness -l loudness1s \
    --val_num_chunks 100
```

### Important arguments

- `--input_path`: folder containing the source audio files
- `--output_path`: LMDB output directory for the training split
- `--config`: dataset gin config from `platune/configs/datasets/`
- `--emb_model_path`: codec used to produce latents, by default `music2latent`
- `--parser_name`: parser used to enumerate files and metadata, parsers can be set in `platune/datasets/parsers.py`
- `--gpu`: GPU index
- `--num_signal`: chunk size in samples
- `--cut_silences`: skip chunks that are silent
- `--save_waveform`: store waveforms in the LMDB; required if you want to run `prepare_augmented_dataset_v4.py` later
- `--use_basic_pitch`: extract MIDI directly from audio **(Make sure this is enabled for MIDI conditioned genration)**
- `--midi_attributes`: compute MIDI-derived control attributes **(Make sure this is enabled for MIDI conditioned genration)**
- `-l` / `--descriptors_list`: one or more continuous audio descriptors to compute
- `--val_num_chunks`: number of examples moved into a separate validation LMDB

## 2. Prepare the augmented LMDB

`prepare_augmented_dataset_v4.py` reads the base LMDB, applies a chained augmentation pipeline, re-encodes the augmented audio with the codec, and writes the result to a second LMDB.


### Example

```bash
python prepare_augmented_dataset_v4.py \
    --lmdb_path datasets/guitatset-test \
    --output_path datasets/guitatset-test_aug \
    --n_augmented 1200 \
    --db_size 4 \
    --config augmentations_v4 \
    --codec music2latent \
    --gpu 0
```

### Important arguments

- `--lmdb_path`: source LMDB created by `prepare_dataset.py`
- `--output_path`: output LMDB for augmented entries
- `--n_augmented`: number of augmented examples to generate
- `--config`: augmentation gin config from `platune/configs/augmentations/`

## 3. Train the model

`train.py` trains the full PLaTune model using a gin config from `platune/configs/models/`.

### Basic training example

```bash
python train.py \
    --db_path datasets/guitatset-test \
    --name guitatset_full \
    --save_path runs \
    --config mini \
    --gpu 0 \
    --max_steps 300000 \
    --val_every 15000 \
    --val_lmdb_path datasets/guitatset-test_val
```

### Training with an augmented LMDB

```bash
python train.py \
    --db_path datasets/guitatset-test \
    --name guitatset_full_aug \
    --save_path runs \
    --config mini \
    --gpu 0 \
    --max_steps 300000 \
    --val_every 15000 \
    --val_lmdb_path datasets/guitatset-test_val \
    --augmented_lmdb_path datasets/guitatset-test_aug \
    --aug_prob 0.9
```

### Important arguments

- `--db_path`: training LMDB directory
- `--name`: run name used for logs and checkpoints
- `--save_path`: parent directory for outputs
- `--config`: model gin config from `platune/configs/models/` such as `v1` or `mini`
- `--gpu`: GPU index, or `-1` for CPU
- `--max_steps`: maximum number of training steps
- `--val_every`: validation frequency in steps or effective epochs
- `--val_lmdb_path`: explicit validation LMDB; if omitted, the loader looks for `<db_path>_val`
- `--ckpt`: resume from a checkpoint or a run directory containing checkpoints
- `--augmented_lmdb_path`: one or more augmented LMDBs to sample from during training
- `--aug_prob`: substitution probability for each augmented LMDB

### Training outputs

Each run writes outputs under:

```text
runs/<name>/
```

This includes:
- TensorBoard logs
- checkpoints, including `last.ckpt`
- `config.gin` with the operative model and data configuration

## Recommended end-to-end order

```bash
# 1. Build latent train/val LMDBs
python prepare_dataset.py ...

# 2. Optionally build an augmented LMDB
python prepare_augmented_dataset_v4.py ...

# 3. Train the model
python train.py ...
```
