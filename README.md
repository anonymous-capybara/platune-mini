# PLaTune-mini Workflow

This repository contains three main command-line scripts for the full PLaTune workflow:

1. `prepare_dataset.py` builds the base latent LMDB from audio files.
2. `prepare_augmented_dataset_v4.py` optionally creates an augmented LMDB from that base dataset.
3. `train.py` trains the PLaTune model from the prepared LMDBs.

The examples below follow the same flow as `train-audio-latent-diffusion-full.ipynb`.

All commands below assume your current working directory is `platune-mini/`.

## 1. Prepare the base latent dataset

`prepare_dataset.py` reads audio files, splits them into fixed-length chunks, encodes each chunk with a pretrained codec, and writes the results to an LMDB.

It can also:

- save raw waveforms alongside latents
- compute audio descriptors such as `rms`, `integrated_loudness`, and `loudness1s`
- extract MIDI with BasicPitch
- derive MIDI-based attributes such as melody, pitch, octave, dynamics, and instrument
- create a separate validation LMDB
- write dataset statistics files used later during training

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
- `--emb_model_path`: codec used to produce latents, typically `music2latent`
- `--parser_name`: parser used to enumerate files and metadata
- `--gpu`: GPU index, or `-1` for CPU
- `--num_signal`: chunk size in samples
- `--cut_silences`: skip chunks that are silent
- `--save_waveform`: store waveforms in the LMDB; required if you want to run `prepare_augmented_dataset_v4.py` later
- `--use_basic_pitch`: extract MIDI directly from audio
- `--midi_attributes`: compute MIDI-derived control attributes
- `-l` / `--descriptors_list`: one or more continuous audio descriptors to compute
- `--val_num_chunks`: number of examples moved into a separate validation LMDB

### Outputs

Running `prepare_dataset.py` creates:

- `datasets/guitatset-test/`: the training LMDB
- `datasets/guitatset-test_val/`: the validation LMDB when `--val_num_chunks > 0`
- `metadata_attributes.json`: min/max statistics for continuous attributes
- `bins_values.pkl`: quantization bins for continuous attributes
- `lmdb_keys.pkl`: keys for valid examples
- `skip_keys.pkl`: keys skipped due to invalid values
- `config.gin`: the operative dataset processing config

## 2. Prepare the augmented LMDB

`prepare_augmented_dataset_v4.py` reads the base LMDB, applies a chained augmentation pipeline, re-encodes the augmented audio with the codec, and writes the result to a second LMDB.

The available augmentation order is fixed:

1. noise
2. filter
3. pitch shift
4. chorus
5. reverb
6. random mix

Each augmented example stores:

- a new latent tensor `z`
- an `alpha` vector describing the strength of each enabled augmentation
- copied or adjusted attributes
- augmentation metadata

### Example

```bash
python prepare_augmented_dataset_v4.py \
    --lmdb_path datasets/guitatset-test \
    --output_path datasets/guitatset-test_aug \
    --n_augmented 200 \
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
- `--codec`: codec used to re-encode the augmented waveforms
- `--gpu`: GPU index, or `-1` for CPU
- `--db_size`: maximum LMDB size in GB
- `--lmdb_keys_file`: optional subset of source keys to use as anchors
- `--save_waveform`: optionally store augmented waveforms too

### Notes

- The source LMDB must contain waveforms, so the original dataset must have been prepared with `--save_waveform`.
- Pitch-shift augmentation updates pitch-related attributes without rerunning BasicPitch.
- The script writes augmentation details into LMDB metadata so `train.py` can auto-discover the alpha channels.

## 3. Train the model

`train.py` trains the full PLaTune model using a gin config from `platune/configs/models/`.

The training script:

- loads the base train/validation LMDBs
- optionally mixes in one or more augmented LMDBs during training
- auto-discovers alpha-conditioning channels from augmented LMDB metadata
- reads continuous-attribute statistics from `metadata_attributes.json` or `bins_values.pkl`
- writes checkpoints and TensorBoard logs under the chosen run directory

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
- `--lmdb_keys_filename`: optional filtered key list to use from the LMDB
- `--min_max_file`: continuous-attribute statistics file, default `metadata_attributes.json`
- `--bins_values_file`: optional quantized continuous-attribute file, mutually exclusive with `--min_max_file`
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

For a concrete runnable example, see `train-audio-latent-diffusion-full.ipynb`.
