
from pathlib import Path
import os
import torch

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