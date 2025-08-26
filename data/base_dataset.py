from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset


class BaseDataset(Dataset,ABC):
    """Base dataset class for Few-Shot Classification and Segmentation (FS-CS) or 
    Few-Shot Segmentation (FS-S).

    Parameters
    ----------
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
    """
    
    def __init__(
        self, 
        split: str, 
        fold: int, 
        way: int, 
        shot: int, 
        transform: transforms,
        setting: str = "original",
        empty_masks: bool = True
    ) -> None:
        super().__init__()
        self.split = "val" if split in ["val", "test"] else "trn"
        self.nfolds = 4
        self.fold = fold
        self.way = way
        self.shot = shot
        self.transform = transform
        self.setting = setting
        self.empty_masks = empty_masks
        
        # ---- Must be defined by the dataset subclasses  ----
        self.class_ids = None
        self.data = None
        self.data_classwise = None
        
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns
        -------
        int
            Number of items exposed by the dataset.
        """
        
        if self.split == "trn":
            return len(self.data)
        return 1000
    
    # ----------------------------------------------------------------------------------
    # Abstract API - to be implemented by the dataset subclasses
    # ----------------------------------------------------------------------------------

    @abstractmethod
    def __getitem__(self, idx: int) -> None:
        """Return the task for the idx-th query image.

        Parameters
        ----------
        idx : int
            Query image index.
        """
        raise NotImplementedError
    
    @abstractmethod
    def read_img(self, name: str) -> None:
        """Load the specified image and return a PIL image.

        Parameters
        ----------
        name : str
            Image file name.
        """
        raise NotImplementedError

    @abstractmethod
    def read_mask(self, name: str) -> None:
        """Load the specified segmentation mask and return a Torch tensor.

        Parameters
        ----------
        name : str
            Mask file name.
        """
        raise NotImplementedError
    
    # ----------------------------------------------------------------------------------
    # Task sampling and batch creation
    # ----------------------------------------------------------------------------------

    def sample_support_set(
        self, 
        q_name: str, 
        q_class: int
    ) -> Tuple[List[List[str]], List[int]]:
        """Sample a support set for the given query image.

        Parameters
        ----------
        q_name : str
            Image file name of the query image.
        q_class : int
            Ground-truth class of the query image.

        Returns
        -------
        Tuple[List[List[str]], List[int]]
            "(s_names, s_classes)" where:
            * "s_names" - nested list of image file names for the support set of shape 
            "(Way, Shot)".
            * "s_classes" - list of class ids per way of length "Way".
        """
        
        nclass = len(self.class_ids)
        if self.empty_masks:
            sampling_prob = np.ones([nclass]) / 2.0 / float(nclass - 1)
            sampling_prob[self.class_ids.index(q_class)] = 1 / 2.0
            s_classes = np.random.choice(
                self.class_ids, self.way, p=sampling_prob, replace=False
            ).tolist()
        else:
            s_classes = [q_class]
            if self.way > 1:
                possible_classes = self.class_ids.copy()
                possible_classes.remove(q_class)
                s_classes += np.random.choice(
                    possible_classes, self.way-1, replace=False
                ).tolist()
            
        s_names = []
        for i in s_classes:
            s_names_i = []
            possible_shots = self.data_classwise[i]
            while len(s_names_i) != self.shot:  
                s_name = np.random.choice(possible_shots, 1, replace=False)[0]
                if q_name != s_name and s_name not in s_names_i:
                    s_names_i.append(s_name)
            s_names.append(s_names_i)

        return s_names, s_classes

    def get_batch(
        self, 
        q_name: str, 
        s_names: List[List[str]], 
        s_classes: List[int], 
        theta: float
    ) -> Dict[str, Any]:
        """Creates a batch from the specified data.

        Parameters
        ----------
        q_name : str
            Image file name of the query image.
        s_names : List[List[str]]
            Nested list of image file names for the support set of shape "(Way, Shot)".
        s_classes : List[int]
            List of class ids per way of length "Way".
        theta : float
            Area threshold in "[0, 1]" used to filter out any object that occupies less 
            than this threshold when an augmented setting is used ("partially-augmented" 
            or "fully-augmented").

        Returns
        -------
        Dict[str, Any]
            Batch dictionary:
            * "q_img": torch.Tensor - query image of shape "(3, H, W)" where H and W 
            correspond to the IMG_SIZE specified in the configuration file.
            * "q_mask": torch.Tensor - ground-truth mask of shape "(H, W)".
            * "q_classes": torch.Tensor - ground-truth labels of shape "(Way)".
            * "q_name": str - image file name of the query image.
            * "q_img_size": torch.Tensor - original size (H and W) of query image of 
            shape "(2)".
            * "s_imgs": torch.Tensor - support images of shape "(Way, Shot, 3, H, W)".
            * "s_masks": torch.Tensor - support masks of shape "(Way, Shot, H, W)".
            * "s_names": List[List[str]] - nested list of image file names for the 
            support set of shape "(Way, Shot)".
            * "s_classes": torch.Tensor - support classes of shape "(Way)".
            * "task_augmented": bool - flag indicating whether the current task has been 
            augmented.
        """

        q_img, q_mask, s_imgs, s_masks = self.get_task(q_name, s_names)
        q_img_size = torch.tensor(q_img.size)
        
        task_augmented = False
        if self.setting in ["partially-augmented", "fully-augmented"]:
            s_imgs, s_masks, s_classes, task_augmented = self.augment_support_set(
                s_imgs, s_masks, s_classes, theta
            )
        
        q_classes = [c in torch.unique(q_mask) for c in s_classes] 
        rename_class = lambda x: s_classes.index(x) + 1 if x in s_classes else 0
        
        q_img = self.transform(q_img)
        q_mask = self.get_q_mask(q_img, q_mask, rename_class)
        s_imgs = torch.stack(
            [
                torch.stack([self.transform(img) for img in imgs]) 
                for imgs in s_imgs
            ]
        )
        s_masks = self.get_s_masks(s_imgs, s_classes, s_masks, rename_class)

        s_classes = torch.tensor(s_classes)
        q_classes = torch.tensor(q_classes)

        return {
            "q_img": q_img,
            "q_mask": q_mask,
            "q_classes": q_classes,
            "q_name": q_name,
            "q_img_size": q_img_size,
            "s_imgs": s_imgs, 
            "s_masks": s_masks, 
            "s_names": s_names, 
            "s_classes": s_classes,
            "task_augmented": task_augmented
        }
    
    def get_task(self, q_name: str, s_names: List[List[str]]) -> Tuple[
        JpegImageFile, 
        torch.Tensor, 
        List[List[JpegImageFile]],
        List[List[torch.Tensor]]
    ]:
        """Load the specified query and support set.

        Parameters
        ----------
        q_name : str
            Image file name of the query image
        s_names : List[List[str]]
            Nested list of image file names for the support set of shape "(Way, Shot)".

        Returns
        -------
        Tuple[
            PIL.JpegImagePlugin.JpegImageFile, 
            torch.Tensor, 
            List[List[PIL.JpegImagePlugin.JpegImageFile]], 
            List[List[torch.Tensor]]
        ]
            "(q_img, q_mask, s_imgs, s_masks)" where:
            * "q_img" - query image.
            * "q_mask" - query mask.
            * "s_imgs" - nested list of support images of shape "(Way, Shot)".
            * "s_imgs" - nested list of support masks of shape "(Way, Shot)".
        """

        q_img  = self.read_img(q_name)
        q_mask = self.read_mask(q_name)
        s_imgs = []
        s_masks = []
        for names in s_names:
            s_imgs_i = []
            s_masks_i = []
            for name in names:
                s_imgs_i.append(self.read_img(name))
                s_masks_i.append(self.read_mask(name))
            s_imgs.append(s_imgs_i) 
            s_masks.append(s_masks_i) 

        return q_img, q_mask, s_imgs, s_masks

    def augment_support_set(
        self,
        imgs: List[List[JpegImageFile]],
        masks: List[List[torch.Tensor]],
        classes: List[int],
        theta: float,
    ) -> Tuple[List[List[JpegImageFile]], List[List[torch.Tensor]], List[int], bool]:
        """Augment the support set with support classes  (partially-augmented) or with 
        both support and non-support classes (fully-augmented).

        Parameters
        ----------
        imgs : List[List[PIL.JpegImagePlugin.JpegImageFile]]
            Nested list of support images of shape "(Way, Shot)"
        masks : List[List[torch.Tensor]]
            Nested list of support masks of shape "(Way, Shot)".
        classes : List[int]
            List of class ids per way of length "Way".
        theta : float
            Area threshold in "[0, 1]" used to filter out any object that occupies less 
            than this threshold.

        Returns
        -------
        Tuple[
            List[List[PIL.JpegImagePlugin.JpegImageFile]], 
            List[List[torch.Tensor]], 
            List[int], 
            bool
        ]
            "(imgs, masks, classes, augmented)" where:
            * "imgs" - augmented support images.
            * "masks" - augmented support masks.
            * "classes" - augmented support classes.
            * "augmented" - flag indicating whether the current task has been augmented.
        """
        
        target_classes = (
            classes if self.setting == "partially-augmented" 
            else self.class_ids
        )
        rename_class = {c: i for i, c in enumerate(classes)}
        ignore_classes = (0, 255) 

        task_augmented = False
        for w in range(self.way):
            class_id = classes[w]
            for s in range(self.shot):
                img = imgs[w][s]
                mask = masks[w][s]
                m_classes, class_pixels = torch.unique(mask, return_counts=True)
                total_pixels = mask.numel()

                for c, pixels in zip(m_classes.tolist(), class_pixels.tolist()):
                    if c in ignore_classes or c == class_id or c not in target_classes:
                        continue
                    if pixels / total_pixels <= theta:
                        continue

                    task_augmented = True
                    if self.setting == "fully-augmented" and c not in rename_class:
                        rename_class[c] = len(classes)
                        classes.append(c)
                        imgs.append([])
                        masks.append([])

                    imgs[rename_class[c]].append(img)
                    masks[rename_class[c]].append(mask)

        if not task_augmented:
            return imgs, masks, classes, task_augmented

        max_shots = max(len(shots) for shots in imgs)
        for w in range(len(imgs)):
            while len(imgs[w]) != max_shots:
                idx = np.random.randint(len(imgs[w]))
                imgs[w].append(imgs[w][idx])
                masks[w].append(masks[w][idx])

        return imgs, masks, classes, task_augmented

    def get_q_mask(
        self, 
        img: torch.Tensor, 
        mask: torch.Tensor, 
        rename_class: Callable[[int], int]
    ) -> torch.Tensor:
        """Update the query mask to remove non-support classes and add the background 
        class.

        Parameters
        ----------
        img : torch.Tensor
            Query image of shape "(3, H, W)" where H and W correspond to the IMG_SIZE 
            specified in the configuration file.
        mask : torch.Tensor
            Ground-truth mask.
        rename_class : Callable[[int], int]
            Mapping function of the support classes.

        Returns
        -------
        torch.Tensor
            Updated ground-truth mask.
        """
        
        if self.split == "trn":
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), 
                img.size()[-2:], 
                mode="nearest"
            ).squeeze()
        
        q_mask = torch.zeros_like(mask).to(mask.device).type(mask.dtype)
        classes = torch.unique(mask)
        for c in classes:
            q_mask[mask == c] = 0 if c in [0, 255] else rename_class(c)
        return q_mask

    def get_s_masks(
        self, 
        imgs: torch.Tensor, 
        classes: List[int], 
        masks: List[List[torch.Tensor]], 
        rename_class: Callable[[int], int]
    ) -> torch.Tensor:
        """Update the support masks to remove non-support classes and add the background 
        class.

        Parameters
        ----------
        imgs : torch.Tensor
            Support images of shape "(Way, Shot, 3, H, W)" where H and W correspond to 
            the IMG_SIZE specified in the configuration file.
        classes : List[int]
            Support classes of shape "(Way)".
        masks : List[List[torch.Tensor]]
            Support masks.
        rename_class : Callable[[int], int]
            Mapping function of the support classes.

        Returns
        -------
        torch.Tensor
            Updated support masks.
        """
        
        s_masks = []
        for c, masks_i in zip(classes, masks): 
            s_masks_i = []
            for mask in masks_i:
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(), 
                    imgs.size()[-2:], 
                    mode="nearest"
                ).squeeze()
                mask[mask != c] = 0
                mask[mask == c] = rename_class(c)
                s_masks_i.append(mask)
            s_masks.append(torch.stack(s_masks_i))
        s_masks = torch.stack(s_masks)
        return s_masks
