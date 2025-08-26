from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing_extensions import override

import numpy as np
import PIL.Image as Image
import torch
from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms

from data.base_dataset import BaseDataset


class DatasetPASCAL(BaseDataset):
    """PASCAL-5i dataset for Few-Shot Classification and Segmentation (FS-CS) or 
    Few-Shot Segmentation (FS-S).

    Parameters
    ----------
    data_path : str
        Path to the directory that contains the dataset.
    split : str
        Dataset split in {"trn", "val", "test"}.
    fold : int
        Fold index in {0, 1, 2, 3}.
    way : int
        Number of classes per task.
    shot : int
        Number of support images per class.
    transform : torchvision.transforms.Compose
        Transform pipeline applied to images.
    setting : str, default="original"
        Evaluation settin in {"original", "partially-augmented", "fully-augmented"}.
    empty_masks : bool, default=True
        If "True", query images may contain no support class.
    **kwargs : Any
        Present for compatibility across datasets.
    """
    
    def __init__(
        self, 
        data_path: str,
        split: str, 
        fold: int, 
        way: int, 
        shot: int, 
        transform: transforms,
        setting: str = "original",
        empty_masks: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            split, fold, way, shot, transform, setting, empty_masks
        )
        self.name = "pascal"
        self.nclass = 20

        root = Path(data_path)
        self.imgs_path = root / "PASCAL" / "JPEGImages"
        self.masks_path = root / "PASCAL" / "SegmentationClassAug"
        self.class_ids = self.get_class_ids()
        self.data = self.get_data()
        self.data_classwise = self.get_data_classwise()
        
    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the task for the idx-th query image.

        Parameters
        ----------
        idx : int
            Query image index.
            
        Returns
        -------
        Dict[str, Any]
            Batch dictionary.
        """
        
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.data)
        q_name, q_class = self.data[idx]
        s_names, s_classes = self.sample_support_set(q_name, q_class)
        return self.get_batch(q_name, s_names, s_classes, 0.07)
    
    @override
    def read_img(self, name: str) -> JpegImageFile:
        """Load the specified image.

        Parameters
        ----------
        name : str
            Image file name.
            
        Returns
        -------
        PIL.JpegImagePlugin.JpegImageFile
            The loaded image in RGB mode.
        """
        
        return Image.open(self.imgs_path / f"{name}.jpg")

    @override
    def read_mask(self, name: str) -> torch.Tensor:
        """Load the specified segmentation mask.

        Parameters
        ----------
        name : str
            Mask file name.
            
        Returns
        -------
        torch.Tensor
            The loaded mask.
        """
        
        return torch.tensor(np.array(Image.open(self.masks_path / f"{name}.png")))
    
    def get_class_ids(self) -> List[int]:
        """Return class IDs of the current split and fold.

        Returns
        -------
        List[int]
            List of class IDs.
        """
        
        nclass_val = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_val + i for i in range(1, nclass_val + 1)]
        if self.split == "trn":
            class_ids_trn = [
                x 
                for x in range(1, self.nclass + 1) 
                if x not in class_ids_val
            ]
            return class_ids_trn
        return class_ids_val
    
    def get_data(self) -> List[Tuple[str, int]]:
        """Load the list of "[image_id, class_id]" pairs for the current split and fold.

        Returns
        -------
        List[Tuple[str, int]]
            List of "[image_id, class_id]" pairs.
        """
        
        if self.split == "trn":
            data = []
            for fold in range(self.nfolds):
                if fold == self.fold: 
                    continue
                data += self.read_data(fold)
            return data
        return self.read_data(self.fold)
    
    def get_data_classwise(self) -> Dict[int, List[str]]:
        """Groups the images by class.

        Returns
        -------
        Dict[int, List[str]]
            Mapping from each class ID to a list of image names belonging to that class.
        """
        
        data_classwise = defaultdict(list)
        for img_name, img_class in self.data:
            data_classwise[img_class] += [img_name]
        return data_classwise
    
    def read_data(self, fold: int) -> List[Tuple[str, int]]:
        """Read the split file for a specific fold and parse its information.

        Parameters
        ----------
        fold : int
            Fold index in {0, 1, 2, 3}.

        Returns
        -------
        List[Tuple[str, int]]
            List of "[image_id, class_id]" pairs.
        """

        file = Path("data") / "splits" / "pascal" / self.split / f"fold{fold}.txt"
        return [
            (data.split("__")[0], int(data.split("__")[1])) 
            for data in file.read_text().splitlines()
        ]
