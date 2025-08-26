from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
from typing_extensions import override

import numpy as np
import PIL.Image as Image
import torch
from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms

from data.base_dataset import BaseDataset


class DatasetCOCOCustom(BaseDataset):
    """COCO-20i dataset for Few-Shot Classification and Segmentation (FS-CS) or 
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
        self.name = "coco"
        if self.split != "val":
            raise Exception(f"Only 'val' split is possible, got: '{self.split}'")
        self.nclass = 80
        self.object_size = object_size
        
        self.base_path = Path(data_path) / "COCO"
        (
            self.class_ids, 
            self.data_classwise, 
            self.data_foldwise
        ) = self.get_data_class_and_foldwise()  
        self.data, self.samples_classwise, self.max_tasks = self.get_data()

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
    
    def get_data_class_and_foldwise(self) -> Tuple[
        List[int],
        Dict[int, List[str]], 
        Dict[int, Dict[int, List[str]]]
    ]:
        """Load all data grouped by class and by fold.

        Returns
        -------
        Tuple[List[int], Dict[int, List[str]], Dict[int, Dict[int, List[str]]]]
            "(class_ids, data_classwise, data_foldwise)" where:
            * "class_ids" - list of class IDs.
            * "data_classwise" - mapping from each class ID to a list of image names 
            belonging to that class.
            * "data_foldwise" - mapping from each fold to a dictionary of data grouped 
            by class.
        """
        
        data_foldwise = {}
        for fold in range(self.nfolds):
            data_foldwise[fold] = self.read_data(fold)
        data_classwise = data_foldwise[self.fold]
        return list(data_classwise.keys()), data_classwise, data_foldwise
    
    def get_data(self) -> Tuple[List[Tuple[str, int]], Dict[int, int], int]:
        """Load all relevant data for the current fold.

        Returns
        -------
        Tuple[List[Tuple[str, int]], Dict[int, int], int]
            "(data, samples_classwise, max_tasks)" where:
            * "data" - list of "[image_id, class_id]" pairs.
            * "samples_classwise" - mapping from each class ID to the number of images
            belonging to that class.
            * "max_tasks" - maximum number of images across all folds.
        """
        
        data = []
        samples_classwise = {}
        max_tasks = 0
        for fold in range(self.nfolds):
            data_f, samples_classwise_f = self.filter_data(self.data_foldwise[fold])
            max_tasks = max(max_tasks, len(data_f))
            if fold == self.fold:
                data = data_f
                samples_classwise = samples_classwise_f
             
        np.random.shuffle(data)
        return data, samples_classwise, max_tasks
    
    def read_data(self, fold: int) -> Dict[int, List[str]]:
        """Read the file for a specific fold and parse its information.

        Parameters
        ----------
        fold : int
            Fold index in {0, 1, 2, 3}.

        Returns
        -------
        Dict[int, List[str]]
            Mapping from each class ID to a list of image names belonging to that class.
        """
        
        data = {}
        file_name = f"./data/splits/coco/custom/{self.object_size}/fold{fold}.pkl"
        with open(file_name, "rb") as f:
            data_fold = pickle.load(f)
            for k, v in data_fold.items():
                if len(v) > self.shot:
                    data[k + 1] = v
        return data
    
    def filter_data(
        self, 
        data_classwise: Dict[int, List[str]]
    ) -> Tuple[List[Tuple[str, int]], Dict[int, int]]:
        """Filter the specified data and count the number of available examples per 
        class.

        Parameters
        ----------
        data_classwise : Dict[int, List[str]]
            Mapping from each class ID to a list of image names belonging to that class.
            
        Returns
        -------
        Tuple[List[Tuple[str, int]], Dict[int, int]]
            "(filtered_data, samples_classwise)" where:
            * "filtered_data" - filtered list of "[image_id, class_id]" pairs.
            * "samples_classwise" - mapping from each class ID to the number of images
            belonging to that class.
        """
        
        samples_classwise = {k: len(v) for k, v in data_classwise.items()}
        filtered_data = []
        for c in data_classwise.keys():
            if self.way == 1 or samples_classwise[c] > self.shot:
                for img_name in data_classwise[c]:
                    filtered_data.append([img_name, c])
        return filtered_data, samples_classwise 
    