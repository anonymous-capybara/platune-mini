import json
import os
from pathlib import Path

import click
import torch
import pytorch_lightning as pl

from platune.datasets.dataset import load_data
from platune.model_mini import PLaTuneMini


torch.set_float32_matmul_precision("high")


def search_for_run(run_path, mode="last"):
    if run_path is None:
        return None
    if ".ckpt" in run_path:
        return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in os.path.basename(str(e)), ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts):
        return ckpts[-1]
    return None


def select_accelerator(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        return "cuda", [gpu]
    if torch.mps.is_available() and gpu >= 0:
        return "mps", 1
    return "cpu", 1


@click.command()
@click.option("-d", "--db_path", default="", help="Dataset path")
@click.option("--val_lmdb_path", default=None, help="Validation LMDB path")
@click.option("-n", "--name", default="platune_mini", help="Run name")
@click.option("-s", "--save_path", default="runs", help="Checkpoint/log output path")
@click.option("--gpu", default=-1, help="GPU to use")
@click.option("--ckpt", default=None, help="Path to a checkpoint or run directory")
@click.option("--max_steps", default=300_000, help="Maximum training steps")
@click.option("--val_every", default=10_000, help="Validation frequency in steps")
@click.option("--build_cache", is_flag=True, help="Load the dataset into memory")
@click.option("--lmdb_keys_filename", default="lmdb_keys", help="LMDB keys filename")
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--n_workers", default=7, help="Number of dataloader workers")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--nb_steps", default=20, help="Number of ODE integration steps")
@click.option("--n_audio_examples", default=12, help="Validation audio clips to log")
def main(
    db_path,
    val_lmdb_path,
    name,
    save_path,
    gpu,
    ckpt,
    max_steps,
    val_every,
    build_cache,
    lmdb_keys_filename,
    batch_size,
    n_workers,
    lr,
    nb_steps,
    n_audio_examples,
):
    train_loader, val_loader, _ = load_data(
        data_path=db_path,
        discrete_keys=[],
        continuous_keys=[],
        batch_size=batch_size,
        n_workers=n_workers,
        cache=build_cache,
        lmdb_keys_file=lmdb_keys_filename,
        augmented_lmdb_paths=None,
        n_alpha_channels=0,
        val_data_path=val_lmdb_path,
    )

    model_config = PLaTuneMini.default_model_config()
    model_config.update({
        "lr": lr,
        "nb_steps": nb_steps,
        "n_audio_examples": n_audio_examples,
    })
    model = PLaTuneMini(**model_config)

    run_dir = os.path.join(save_path, name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "cli_args": {
            "db_path": db_path,
            "val_lmdb_path": val_lmdb_path,
            "name": name,
            "save_path": save_path,
            "gpu": gpu,
            "ckpt": ckpt,
            "max_steps": max_steps,
            "val_every": val_every,
            "build_cache": build_cache,
            "lmdb_keys_filename": lmdb_keys_filename,
            "batch_size": batch_size,
            "n_workers": n_workers,
            "lr": lr,
            "nb_steps": nb_steps,
            "n_audio_examples": n_audio_examples,
        },
        "model_config": model_config,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as config_out:
        json.dump(config, config_out, indent=2)

    callbacks = [pl.callbacks.ModelCheckpoint(filename="last")]

    val_check = {}
    if val_loader is not None and len(train_loader) > 0:
        if len(train_loader) >= val_every:
            val_check["val_check_interval"] = val_every
            print(f"Validation will be checked every {val_every} training steps.")
        else:
            n_epoch = max(1, val_every // len(train_loader))
            val_check["check_val_every_n_epoch"] = n_epoch
            print(f"Validation will be checked every {n_epoch} epochs.")

    accelerator, devices = select_accelerator(gpu)
    print(f"device - selected gpu: {accelerator}:{devices}")

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(save_path, name=name),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=100000,
        max_steps=max_steps,
        profiler="simple",
        enable_progress_bar=True,
        **val_check,
    )

    run = search_for_run(ckpt)
    if run is not None:
        print(f"Restarting from checkpoint: {run}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=run)


if __name__ == "__main__":
    main()
