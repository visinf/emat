import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from yacs.config import CfgNode

from data.builder import dataset_builder


class DataModule(pl.LightningDataModule):
    """Data module for training, validation, and testing of Few-Shot Classification and 
    Segmentation (FS-CS) or Few-Shot Segmentation (FS-S) methods.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Experiment configuration.
    args : argparse.Namespace
        Arguments providing data and task configuration, and runtime flags.
    """

    def __init__(self, cfg: CfgNode, args: argparse.Namespace) -> None:
        super().__init__()
        self.cfg = cfg
        self.args = args
        resize = transforms.Resize(size=(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE))
        tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([resize, tensor, normalize])

    def setup(self, stage: str = None) -> None:
        """Instantiate datasets for the requested stage.

        Parameters
        ----------
        stage : str
            Execution stage in {"fit", "test"}
        """
        
        dataset = self.cfg.DATA.DATASET
        if self.args.object_size_split is not None:
            dataset += "-custom-object-size"
        builder = dataset_builder[dataset]
        
        if stage == "fit":
            self.train_dataset = builder(
                data_path=self.cfg.DATA.PATH, 
                split="trn",
                fold=self.args.fold,
                way=1,
                shot=1,
                transform=self.transform,
                setting="original",
                empty_masks=self.args.empty_masks
            )
            self.val_dataset = builder(
                data_path=self.cfg.DATA.PATH, 
                split="val",
                fold=self.args.fold,
                way=self.args.way,
                shot=self.args.shot,
                transform=self.transform,
                setting=self.args.setting,
                empty_masks=self.args.empty_masks
            )
        
        elif stage == "test":
            self.test_dataset = builder(
                data_path=self.cfg.DATA.PATH, 
                split="test",
                fold=self.args.fold,
                way=self.args.way,
                shot=self.args.shot,
                transform=self.transform,
                setting=self.args.setting,
                empty_masks=self.args.empty_masks,
                object_size=self.args.object_size_split
            )
            
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            Training dataloader.
        """
        
        return self.init_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            Validation dataloader.
        """
        
        return self.init_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the testing dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            Testing dataloader.
        """
        
        return self.init_dataloader(self.test_dataset, shuffle=False)
        
    def init_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Initializes a dataloader for the specified dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to be wrapped in the dataloader.
        shuffle : bool, default=True
            Whether to shuffle the data during loading.

        Returns
        -------
        torch.utils.data.DataLoader
            Configured dataloader.
        """
        
        g = torch.Generator()
        g.manual_seed(self.cfg.TRAIN.SEED)
        workers_flag = self.cfg.TRAIN.DATA_LOADER.NUM_WORKERS > 0
        
        return DataLoader(
            dataset, 
            batch_size=self.cfg.TRAIN.DATA_LOADER.BATCH_SIZE if shuffle else 1,
            shuffle=shuffle, 
            num_workers=self.cfg.TRAIN.DATA_LOADER.NUM_WORKERS,
            pin_memory=workers_flag,
            persistent_workers=workers_flag,
            generator=g
        )
