from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class Evaluator:
    """Aggregate metrics (accuracy, mean-IoU, and loss) per step.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset instance:
        * "name" - dataset alias (e.g., "pascal" or "coco").
        * "class_ids" - IDs of the target classes.
    """
    
    def __init__(self, dataset: Dataset) -> None:
        self.nsamples = 0
        self.ncorrect_samples = 0
        self.cumulative_loss = 0.0
        self.loss_steps = 0
        
        total_classes = 20 if dataset.name == "pascal" else 80
        self.area_inter = torch.zeros(total_classes + 1, dtype=torch.float32)
        self.area_union = torch.zeros_like(self.area_inter, dtype=torch.float32)
        
        self.class_ids = torch.as_tensor(dataset.class_ids, dtype=torch.long)
        self._ones = torch.ones(self.class_ids.numel(), dtype=torch.float32)

    # ----------------------------------------------------------------------------------
    # Update helpers (called per step)
    # ----------------------------------------------------------------------------------
    
    def update_clf(self, pred_labels: Tensor, batch: Dict[str, Tensor]) -> None:
        """Accumulate correct classification predictions.

        Parameters
        ----------
        pred_labels : torch.Tensor
            Predicted labels of shape "(B, Way)" where B is the batch size.
        batch : Dict[str, torch.Tensor]
            Batch dictionary:
            * "q_classes" - ground-truth labels of shape "(B, Way)".
        """
        
        correct_preds = (pred_labels == batch["q_classes"]).all(dim=1)
        self.ncorrect_samples += int(correct_preds.sum())

    def update_seg(self, pred_masks: Tensor, batch: Dict[str, Tensor]) -> None:
        """Update intersection and union statistics for mIoU computation.

        Parameters
        ----------
        pred_masks : torch.Tensor
            Predicted segmentation masks of shape "(B, H, W)" where B is the batch size
            and H and W are the spatial dimensions of the query image.
        batch : Dict[str, torch.Tensor]
            Batch dictionary:
            * "q_mask" - ground-truth masks of shape "(B, H, W)".
            * "s_classes" - support classes of shape "(B, Way)".
        """
        
        pred_masks = pred_masks.cpu()
        q_mask = batch["q_mask"].to(torch.int64).cpu()
        s_classes = batch["s_classes"].cpu()

        bsz = s_classes.size(0)
        self.nsamples += bsz
        
        bg_class = torch.zeros(bsz, 1, dtype=torch.int64)
        s_classes = torch.cat([bg_class, s_classes], dim=1)
        
        for classes, p_mask, gt_mask in zip(s_classes, pred_masks, q_mask):
            way = classes.numel() - 1
            area_inter, area_union = self.intersect_and_union(p_mask, gt_mask, way)
            self.area_inter.scatter_add_(0, classes, area_inter)
            self.area_union.scatter_add_(0, classes, area_union)
            
    def update_loss(self, loss: float) -> None:
        """Accumulate loss.

        Parameters
        ----------
        loss : float
            Step loss value.
        """
        
        self.cumulative_loss += loss
        self.loss_steps += 1

    # ----------------------------------------------------------------------------------
    # Metric computation (call at end of epoch)
    # ----------------------------------------------------------------------------------

    def compute_avg_acc(self) -> float:
        """Compute the per-epoch average accuracy in percentage."""
        
        if self.nsamples == 0:
            return 0.0
        return 100.0 * self.ncorrect_samples / self.nsamples

    def compute_miou(self) -> float:
        """Compute the per-epoch mean intersection-over-union in percentage."""
        
        inter = self.area_inter[self.class_ids] 
        union = self.area_union[self.class_ids]
        iou = inter / torch.maximum(union, self._ones)
        return float(iou.mean() * 100.0)

    def compute_avg_loss(self) -> float:
        """Compute the per-epoch average loss."""
        
        if self.loss_steps == 0:
            return 0.0
        return self.cumulative_loss / self.loss_steps

    # ----------------------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------------------
    
    @staticmethod
    def intersect_and_union(
        pred_mask: Tensor, 
        gt_mask: Tensor, 
        way: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute intersection and union via histogram binning.

        Parameters
        ----------
        pred_mask : torch.Tensor
            Predicted mask of shape "(H, W)" where H and W are the spatial dimensions of 
            the query image. The mask contains values in "{0, ..., way}" with 0 
            indicating background.
        gt_mask : torch.Tensor
            Ground-truth mask of shape "(H, W)".
        way : int
            Number of classes.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            "(area_inter, area_union)" - tensors of shape "way + 1".
        """
        
        inter = pred_mask[pred_mask == gt_mask]
        area_inter = torch.histc(inter.float(), bins=way + 1, min=0, max=way)
        area_pred = torch.histc(pred_mask.float(), bins=way + 1, min=0, max=way)
        area_gt = torch.histc(gt_mask.float(), bins=way + 1, min=0, max=way)
        return area_inter, area_pred + area_gt - area_inter
