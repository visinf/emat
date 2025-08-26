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


class DatasetPASCALCustom(BaseDataset):
    """PASCAL-5i dataset for Few-Shot Classification and Segmentation (FS-CS) or 
    Few-Shot Segmentation (FS-S) using objects of a specific size.

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
    object_size : str, default=None
        Specific object size in {"0-5", "5-10", "10-15", "0-15"}.
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
        object_size: str = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            split, fold, way, shot, transform, setting="original", empty_masks=False
        )
        self.name = "pascal"
        if self.split != "val":
            raise Exception(f"Only 'val' split is possible, got: '{self.split}'")
        self.nclass = 20
        self.object_size = object_size

        root = Path(data_path)
        self.imgs_path = root / "PASCAL" / "JPEGImages"
        self.masks_path = root / "PASCAL" / "SegmentationClassAug"
        self.data_foldwise = self.get_data_foldwise()
        (
            self.class_ids, 
            self.data, 
            self.data_classwise, 
            self.samples_classwise, 
            self.max_tasks
        ) = self.get_data()
        
    @override
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            Number of items exposed by the dataset.
        """
        
        return min(1000, self.max_tasks)

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
    
    def get_data_foldwise(self) -> Dict[int, List[Tuple[str, int]]]:
        """Load all data grouped by fold.

        Returns
        -------
        Dict[int, List[Tuple[str, int]]]
            Mapping from each fold to a list of "[image_id, class_id]" pairs.
        """
        
        data_foldwise = {}
        for fold in range(self.nfolds):
            data_foldwise[fold] = self.read_data(fold)
        return data_foldwise
    
    def get_data(self) -> Tuple[
        List[int], 
        List[Tuple[str, int]], 
        Dict[int, List[str]], 
        Dict[int, int], 
        int
    ]:
        """Load all relevant data for the current fold.

        Returns
        -------
        Tuple[
            List[int], 
            List[Tuple[str, int]], 
            Dict[int, List[str]], 
            Dict[int, int], 
            int
        ]
            "(class_ids, data, data_classwise, samples_classwise, max_tasks)" where:
            * "class_ids" - list of class IDs.
            * "data" - list of "[image_id, class_id]" pairs.
            * "data_classwise" - mapping from each class ID to a list of image names 
            belonging to that class.
            * "samples_classwise" - mapping from each class ID to the number of images
            belonging to that class.
            * "max_tasks" - maximum number of images across all folds.
        """
        
        class_ids = []
        data = []
        data_classwise = {}
        samples_classwise = {}
        max_tasks = 0
        for fold in range(self.nfolds):
            (
                data_f,
                data_classwise_f, 
                samples_classwise_f 
            ) = self.group_data_classwise(self.data_foldwise[fold])
            max_tasks = max(max_tasks, len(data_f))
            if fold == self.fold:
                class_ids = list(data_classwise_f.keys())
                data = data_f
                data_classwise = data_classwise_f
                samples_classwise = samples_classwise_f
        
        np.random.shuffle(data)
        return class_ids, data, data_classwise, samples_classwise, max_tasks

    def read_data(self, fold: int) -> List[Tuple[str, int]]:
        """Read the file for a specific fold and parse its information.

        Parameters
        ----------
        fold : int
            Fold index in {0, 1, 2, 3}.

        Returns
        -------
        List[Tuple[str, int]]
            List of "[image_id, class_id]" pairs.
        """
        
        file = (
            Path("data") / 
            "splits" / 
            "pascal" / 
            "custom" / 
            self.object_size / 
            f"fold{fold}.txt"
        )
        return [
            (data.split("__")[0], int(data.split("__")[1])) 
            for data in file.read_text().splitlines()
        ]
    
    def group_data_classwise(
        self, 
        data: List[Tuple[str, int]]
    ) -> Tuple[List[Tuple[str, int]], Dict[int, List[str]], Dict[int, int]]:
        """Group data by class, filter it, and count the number of available examples 
        per class.

        Parameters
        ----------
        data : List[Tuple[str, int]]
            List of "[image_id, class_id]" pairs.

        Returns
        -------
        Tuple[List[Tuple[str, int]], Dict[int, List[str]], Dict[int, int]]
            "(filtered_data, data_classwise, samples_classwise)" where:
            * "filtered_data" - filtered list of "[image_id, class_id]" pairs.
            * "data_classwise" - mapping from each class ID to a list of image names 
            belonging to that class.
            * "samples_classwise" - mapping from each class ID to the number of images
            belonging to that class.
        """
        
        data_classwise = defaultdict(list)
        samples_classwise = defaultdict(int)
        for img_name, img_class in data:
            data_classwise[img_class].append(img_name)
            samples_classwise[img_class] += 1
        
        valid_classes = [
            c 
            for c in data_classwise.keys() 
            if self.way == True or samples_classwise[c] > self.shot
        ]

        filtered_data = [
            [img_name, c]
            for c in valid_classes
            for img_name in data_classwise[c]
        ]

        return filtered_data, data_classwise, samples_classwise
