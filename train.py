import gin
import os
import click
import json
import lmdb
import pickle
import torch
import pytorch_lightning as pl
from pathlib import Path

from platune.datasets.dataset import load_data
from platune.datasets.audio_example import AudioExample
from platune.model import PLaTune


# BINS_VALUES = 'bins_values.pkl'
# MIN_MAX = 'metadata_attributes.json'  

torch.set_float32_matmul_precision('high')

def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): return ckpts[-1]
    else: return None


@click.command()
@click.option('-d', '--db_path', default="", help='dataset path')
@click.option('-n', '--name', default="", help='Name of the run')
@click.option('-c', '--config', default="v1", help='Name of the gin configuration file to use')
@click.option('-s', '--save_path', default="", help='path to save models checkpoints')
@click.option('--max_steps', default=300_000, help='Maximum number of training steps')
@click.option('--val_every', default=10_000, help='Checkpoint model every n steps')
@click.option('--gpu', default=-1, help='GPU to use')
@click.option('--ckpt', default=None, help='Path to previous checkpoint of the run')
@click.option('--build_cache', is_flag=True, help='wether to load dataset in cache memory for training')
@click.option('--lmdb_keys_filename', default="lmdb_keys", help='lmdb keys filename')
@click.option('--bins_values_file', default=None, help='path to bins_values pkl file to quantize continuous attributes')
@click.option( '--min_max_file', default="metadata_attributes.json", help='path to metadata_attributes.json for continuous attributes')
@click.option('--val_lmdb_path', default=None, help='path to validation LMDB (overrides auto-discovered <db_path>_val)')
@click.option('--aug_prob', default=None, type=float, multiple=True, help='Per-LMDB probability of substituting training batch (one per --augmented_lmdb_path). Remainder = non-augmented probability.')
@click.option('--augmented_lmdb_path', default=None, multiple=True, help='Path(s) to precomputed augmented LMDB(s). Can be specified multiple times.')
def main(
        db_path, 
        name, 
        config, 
        save_path, 
        max_steps, 
        val_every,     
        gpu, 
        ckpt, 
        build_cache, 
        lmdb_keys_filename,
        bins_values_file, 
        min_max_file,
        val_lmdb_path,
        aug_prob,
        augmented_lmdb_path,
    ):
    
    # load config
    config_file = os.path.join(
        os.path.dirname(__file__),
        "platune",
        "configs",
        "models",
        f"{config}.gin",
    )
    print('loading config file : ', config_file)
    gin.parse_config_files_and_bindings([config_file], []) 

    # Canonical augmentation ordering (must match prepare_augmented_dataset_v4.py)
    AUGMENTATION_NAMES = ["noise", "filter", "pitch_shift", "chorus", "reverb", "mix"]

    # Normalise CLI tuples: Click multiple=True yields empty tuple when not provided
    augmented_lmdb_paths = list(augmented_lmdb_path) if augmented_lmdb_path else []
    aug_probs = list(aug_prob) if aug_prob else []

    # Validate lengths
    if len(aug_probs) > 0 and len(aug_probs) != len(augmented_lmdb_paths):
        raise ValueError(
            f"--aug_prob count ({len(aug_probs)}) must match "
            f"--augmented_lmdb_path count ({len(augmented_lmdb_paths)})")
    if len(aug_probs) > 0 and sum(aug_probs) > 1.0 + 1e-6:
        raise ValueError(
            f"Sum of --aug_prob values ({sum(aug_probs):.4f}) must be <= 1.0")
    # Default: equal split leaving 10% for non-augmented, or 0.5 for single LMDB
    if len(aug_probs) == 0 and len(augmented_lmdb_paths) > 0:
        if len(augmented_lmdb_paths) == 1:
            aug_probs = [0.5]
        else:
            share = 0.9 / len(augmented_lmdb_paths)
            aug_probs = [share] * len(augmented_lmdb_paths)
        print(f"No --aug_prob provided, defaulting to {aug_probs}")

    # Auto-discover alpha channels from all augmented LMDBs
    # Build the canonical union of enabled_augmentations and per-LMDB mappings
    enable_alpha = gin.query_parameter('%ENABLE_ALPHA')
    n_alpha_channels = 0
    augmented_alpha_mappings = []  # List[List[int]]: per-LMDB index mapping into union vector
    per_lmdb_enabled = []          # List[List[str]]: per-LMDB enabled augmentation names
    if enable_alpha and len(augmented_lmdb_paths) > 0:
        union_set = set()
        for path in augmented_lmdb_paths:
            env = lmdb.open(path, lock=False, readonly=True,
                            readahead=False, map_async=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                cursor.first()
                _, first_val = cursor.item()
                ae = AudioExample(first_val)
                meta = ae.get_metadata()
                enabled_augs = meta.get("enabled_augmentations", [])
            env.close()
            per_lmdb_enabled.append(enabled_augs)
            union_set.update(enabled_augs)
            print(f"  LMDB {path}: enabled_augmentations={enabled_augs}")

        # Build canonical union in fixed order
        alpha_union = [name for name in AUGMENTATION_NAMES if name in union_set]
        n_alpha_channels = len(alpha_union)

        # Build per-LMDB mapping: local index -> position in union vector
        for enabled_augs in per_lmdb_enabled:
            mapping = [alpha_union.index(name) for name in enabled_augs]
            augmented_alpha_mappings.append(mapping)

        print(f"Alpha union: {alpha_union} (n_alpha_channels={n_alpha_channels})")
        for i, (path, mapping) in enumerate(zip(augmented_lmdb_paths, augmented_alpha_mappings)):
            print(f"  LMDB {i}: {path} -> mapping {mapping}")
    elif not enable_alpha and len(augmented_lmdb_paths) > 0:
        print("ENABLE_ALPHA=False: augmented LMDBs will be used for training data "
              "but alpha conditioning is disabled (n_alpha_channels=0)")

    # load data
    with gin.unlock_config():
        gin.bind_parameter("dataset.load_data.data_path", db_path)
        gin.bind_parameter("dataset.load_data.n_alpha_channels", n_alpha_channels)
        if len(augmented_lmdb_paths) > 0:
            gin.bind_parameter("dataset.load_data.augmented_lmdb_paths", augmented_lmdb_paths)
            gin.bind_parameter("dataset.load_data.augmented_alpha_mappings", augmented_alpha_mappings)
        if build_cache:
            gin.bind_parameter("dataset.load_data.cache", build_cache)
        if lmdb_keys_filename is not None:
            gin.bind_parameter("dataset.load_data.lmdb_keys_file", lmdb_keys_filename)
        if val_lmdb_path is not None:
            gin.bind_parameter("dataset.load_data.val_data_path", val_lmdb_path)
    train, val, aug_loaders = load_data()

    os.makedirs(os.path.join(save_path, name), exist_ok=True)

    # load min max values / bins continuous descriptors
    continuous_keys = gin.query_parameter('%CONTINUOUS_KEYS')
    min_max_values = []
    bins_values = []
    if len(continuous_keys) > 0:
        if min_max_file is not None and bins_values_file is not None:
            raise ValueError("choose to quantize or not continuous attributes")

        if min_max_file is not None:
            with open(os.path.join(db_path, min_max_file)) as f:
                metadata = json.load(f)

            for k, v in metadata['continuous_attr_min_max'].items():
                if k in continuous_keys:
                    min_max_values.append((v["min"], v["max"]))

        elif bins_values_file is not None:
            with open(os.path.join(db_path, bins_values_file), "rb") as f:
                bins = pickle.load(f)

            for k, v in bins.items():
                if k in continuous_keys:
                    bins_values.append(v)
    print(f"min_max_values: {min_max_values}")
    print(f"bins_values: {bins_values}")
    # instantiate model
    with gin.unlock_config():
        gin.bind_parameter("model.PLaTune.n_alpha_channels", n_alpha_channels)
        if len(min_max_values) > 0:
            gin.bind_parameter(
                "model.PLaTune.min_max_attr_continuous", min_max_values)
        if len(bins_values) > 0:
            gin.bind_parameter("model.PLaTune.bins_values", bins_values)
        if len(aug_probs) > 0:
            gin.bind_parameter("model.PLaTune.aug_probs", aug_probs)
    model = PLaTune()

    # model checkpoints
    callbacks_ckpt = []
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")
    callbacks_ckpt.append(last_checkpoint)
    # Save a separate checkpoint at every validation step (keep all)
    # every_val_checkpoint = pl.callbacks.ModelCheckpoint(
    #     filename="step-{step}",
    #     every_n_train_steps=val_every,
    #     save_top_k=-1,
    # )
    # callbacks_ckpt.append(every_val_checkpoint)

    val_check = {}
    if val is not None:
        if len(train) >= val_every:
            val_check["val_check_interval"] = val_every
            print(f"Validation will be checked every {val_every} training steps.")
        else:
            nepoch = val_every // len(train)
            val_check["check_val_every_n_epoch"] = nepoch
            print(f"Validation will be checked every {nepoch} epochs.")

    # select GPU
    accelerator =  "cuda" if torch.cuda.is_available() and gpu >= 0 else "mps" if torch.mps.is_available() and gpu >= 0 else "cpu"
    if accelerator == "cuda" or accelerator == "mps":
        device = 1
    print(f'device - selected gpu: {accelerator}:{device}')


    # instantiate trainer
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_path, name=name),
        accelerator=accelerator,
        devices=device,
        callbacks=callbacks_ckpt,
        max_epochs=100000,
        max_steps=max_steps,
        profiler="simple",
        enable_progress_bar=True,
        **val_check,
    )

    run = search_for_run(ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        print("Restarting from step : ", step)
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    with open(os.path.join(os.path.join(save_path, name), "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    # Attach augmented DataLoaders to model (used in training_step)
    model._aug_loaders = aug_loaders  # List[DataLoader], may be empty
    model._aug_iters = [None] * len(aug_loaders)

    # train model
    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__":
    main()
