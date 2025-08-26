import pickle
from pathlib import Path
from typing import Any, Dict, List
from typing_extensions import override

import numpy as np
import PIL.Image as Image
import torch
from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms

from data.base_dataset import BaseDataset


class DatasetCOCO(BaseDataset):
    """COCO-20i dataset for Few-Shot Classification and Segmentation (FS-CS) or 
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
        self.name = "coco"
        self.nclass = 80
        
        self.base_path = Path(data_path) / "COCO"
        self.class_ids = self.get_class_ids()
        self.data_classwise = self.get_data_classwise()
        self.data = self.get_data()

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
        
        if self.split == "val":
            np.random.seed(idx)

        q_class = np.random.choice(self.class_ids, 1, replace=False)[0]
        q_name = np.random.choice(self.data_classwise[q_class], 1, replace=False)[0]
        s_names, s_classes = self.sample_support_set(q_name, q_class)
        return self.get_batch(q_name, s_names, s_classes, 0.03)

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
        
        return Image.open(self.base_path / name).convert("RGB")

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
        
        return torch.tensor(
            np.array(
                Image.open(
                    (self.base_path / "annotations" / name).with_suffix(".png")
                )
            )
        )
    
    def get_class_ids(self) -> List[int]:
        """Return class IDs of the current split and fold.

        Returns
        -------
        List[int]
            List of class IDs for the current split and fold.
        """
        
        nclass_val = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v + 1 for v in range(nclass_val)]
        if self.split == "trn":
            class_ids_trn = [
                x 
                for x in range(1, self.nclass + 1) 
                if x not in class_ids_val
            ]
            return class_ids_trn
        return class_ids_val

    def get_data_classwise(self) -> Dict[int, List[str]]:
        """Groups the images by class.

        Returns
        -------
        Dict[int, List[str]]
            Mapping from each class ID to a list of image names belonging to that class.
        """
        
        data_classwise = {}
        with open(f"./data/splits/coco/{self.split}/fold{self.fold}.pkl", "rb") as f:
            img_metadata_classwise_temp = pickle.load(f)
            for k in img_metadata_classwise_temp.keys():
                data_classwise[k + 1] = img_metadata_classwise_temp[k]
        
        return data_classwise

    def get_data(self) -> List[str]:
        """Create the list of image names for the current split and fold.

        Returns
        -------
        List[str]
            List of image names.
        """
        
        data = []
        for v in self.data_classwise.values():
            data += v
        return list(set(data))
