import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from yacs.config import CfgNode

from utils.defaults import get_default_cfg


def check_args(args: argparse.Namespace) -> None:
    """Check that the CLI arguments are valid.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    if args.eval:
        if args.object_size_split is None:
            if args.experiment_path is None:
                raise ValueError(f"'--experiment_path' must be specified.")
            if args.way < 1:
                raise ValueError(f"'--way' must be at least 1.")
            if args.shot < 1:
                raise ValueError(f"'--shot' must be at least 1.")
        else:
            if args.setting != "original":
                raise ValueError(f"'--setting' can only be 'original'.")
            if args.empty_masks:
                raise ValueError(f"'--no_empty_masks' must be included.")
                
    else:
        if args.config_path is None:
            raise ValueError(f"'--config_path' must be specified.")
        if args.way != 1:
            raise ValueError(f"'--way' can only be 1.")
        if args.shot != 1:
            raise ValueError(f"'--shot' can only be 1.")
        if args.setting != "original":
            raise ValueError(f"'--setting' can only be 'original'.")
        if args.object_size_split is not None:
            raise ValueError(f"'--object_size_split' cannot be used.")


def setup(args: argparse.Namespace) -> CfgNode:
    """Set up the configuration for both training a new model or evaluating one.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments providing data and task configuration, and runtime flags.

    Returns
    -------
    yacs.config.CfgNode
        The corresponding non-editable configuration object.
    """
    
    cfg = get_default_cfg()
    
    if args.eval:
        args.experiment_path = Path(args.experiment_path) / f"fold-{args.fold}"
        if not args.experiment_path.exists():
            raise ValueError(f"Experiment not found: {args.experiment_path}")
        cfg.merge_from_file(args.experiment_path / "config.yaml")
        cfg.TRAIN.EPOCHS = 1
        cfg.TRAIN.DATA_LOADER.BATCH_SIZE = 1
    else:
        config_path = Path(args.config_path)
        if not config_path.is_file():
            raise ValueError(f"Configuration file not found: {config_path}")
        cfg.merge_from_file(config_path)
        cfg.CONFIG_PATH = str(config_path)
        
        save_dir = Path(cfg.EXPERIMENT_NAME) / f"fold-{args.fold}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config.yaml").write_text(cfg.dump())
            
    cfg.freeze()
    return cfg


def set_all_seeds(seed: int) -> None:
    """Seed all relevant random-number generators for reproducibility.

    Parameters
    ----------
    seed : int
        Non-negative integer used as the RNG seed. Values are wrapped into the
        32-bit range expected by PyTorch.
    """
    
    if seed < 0:
        raise ValueError("`seed` must be a non‑negative integer.")
    seed = int(seed) % (2 ** 32 - 1)
    
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model: pl.LightningModule) -> Tuple[int, int]:
    """Count trainable and non-trainable parameters of a model.

    Parameters
    ----------
    model : pytorch_lightning.LightningModule
        Model whose parameters are inspected.

    Returns
    -------
    Tuple[int, int]
        "(trainable, non_trainable)".
    """

    trainable, non_trainable = 0, 0 

    for param in model.parameters():
        n = param.numel()
        if param.requires_grad:
            trainable += n
        else:
            non_trainable += n

    return trainable, non_trainable
