import argparse
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataloader import DataModule
from utils.callbacks import MetricsCallback, TBLogger
from models.modules.few_shot_module import FewShotModule
from utils.helpers import check_args, count_parameters, setup, set_all_seeds


def run_experiment(args: argparse.Namespace) -> None:
    """Train or evaluate the model specified by the CLI arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    check_args(args)
    
    cfg = setup(args)
    set_all_seeds(cfg.TRAIN.SEED)
    
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Preparing datasets...")
    datamodule = DataModule(cfg, args)
    
    logging.info("Preparing model...")
    if args.eval:
        ckpt_path = Path(args.experiment_path) / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        logging.info(f"Best model found at epoch: {ckpt['epoch']+1}")
        model = FewShotModule.load_from_checkpoint(
            ckpt_path.as_posix(),
            map_location="cpu",
            strict=True, 
            cfg=cfg, 
            args=args
        )
    else:
        model = FewShotModule(cfg=cfg, args=args)
    
    trn_params, nontrn_params = count_parameters(model)
    logging.info(f"Trainable parameters    : {trn_params:,}")
    logging.info(f"Non-trainable parameters: {nontrn_params:,}")
    logging.info(f"Total parameters        : {trn_params + nontrn_params:,}")
    
    logging.info("Preparing callbacks and logger...")
    metrics_cb = MetricsCallback()
    callbacks=[metrics_cb]
    
    logger = False
    if not args.eval:
        save_dir = f"{cfg.EXPERIMENT_NAME}/fold-{args.fold}"
        ckpt_cb = ModelCheckpoint(
            dirpath=save_dir, 
            filename="best",
            monitor="val/miou",
            mode="max"
        )
        callbacks.append(ckpt_cb)
        logger = TBLogger(save_dir=save_dir, name="logs", version=0)
    
    logging.info("Preparing Trainer...")
    trainer = Trainer(
        accelerator="gpu",
        strategy="auto" if args.eval else DDPStrategy(),
        devices=list(map(int, args.gpus.split(","))),
        callbacks=callbacks,
        max_epochs=cfg.TRAIN.EPOCHS,
        enable_checkpointing=not args.eval,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        logger=logger
    )
    
    if args.eval:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Test FS-SC and FS-S methods.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        help="Path to the config file. Training only."
    )
    parser.add_argument(
        "--experiment_path", 
        type=str, 
        help="Path to the experiment directory. Eval only."
    )
    parser.add_argument(
        "--results_path", 
        type=str, 
        help="If provided, directory to store JSON results. Eval only."
    )
    parser.add_argument(
        "--fold", 
        type=int, 
        default=0, 
        choices=[0, 1, 2, 3], 
        help="Fold index to use."
    )
    parser.add_argument(
        "--way", 
        type=int, 
        default=1, 
        help="Classes per task. Training is 1-way."
    )
    parser.add_argument(
        "--shot", 
        type=int, 
        default=1, 
        help="Support images per class. Training is 1-shot."
    )
    parser.add_argument(
        "--setting", 
        type=str, 
        default="original", 
        choices=["original", "partially-augmented", "fully-augmented"], 
        help="Evaluation setting. Eval only."
    )
    parser.add_argument(
        "--object_size_split", 
        type=str, 
        choices=["0-5", "5-10", "10-15", "0-15"], 
        help="Object size subset. Eval only."
    )
    parser.add_argument(
        "--eval", 
        action="store_true", 
        help="Run evaluation instead of training."
    )
    parser.add_argument(
        "--only_seg", 
        action="store_false", 
        dest="fs_cs", 
        help="Use FS-S (segmentation only) instead of FS-CS (classification and seg)."
    )
    parser.add_argument(
        "--no_empty_masks", 
        action="store_false", 
        dest="empty_masks", 
        help="Disallow empty query masks."
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default="0", 
        help="Comma-separated GPU list, e.g. '0,1'."
    )
    args = parser.parse_args()
    
    run_experiment(args)
