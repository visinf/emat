import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from yacs.config import CfgNode

from models.dinov2.vision_transformer import vit_small
from models.builder import model_builder


class FewShotModule(pl.LightningModule):
    """Module for Few-Shot Classification and Segmentation (FS-CS) or Few-Shot 
    Segmentation (FS-S).

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Experiment configuration.
    args : argparse.Namespace
        Arguments providing data and task configuration, and runtime flags.
    """
    
    def __init__(self, cfg: CfgNode, args: argparse.Namespace) -> None:
        super().__init__()
        # ---- Metadata ----
        self.method = "-".join(cfg.EXPERIMENT_NAME.split("/")[-1].split("-")[:-1])
        self.dataset = cfg.DATA.DATASET
        self.setting = args.setting
        self.way = args.way
        self.shot = args.shot
        self.fold = args.fold
        self.fs_cs = args.fs_cs
        self.empty_masks = args.empty_masks
        self.object_size_split = args.object_size_split
        self.results_path = args.results_path
        
        # ---- Hyperparameters ----
        self.lr = cfg.TRAIN.LEARNING_RATE
        self.loss_balance = 0.1
        self.delta = 0.5
        
        # ---- DINOv2 backbone ----
        self.nlayer = 12
        self.nhead = 6
        self.patch_size = 14 
        self.spatial_dim = cfg.DATA.IMG_SIZE // self.patch_size
        
        self.backbone = vit_small(patch_size=self.patch_size)
        state_dict = torch.load(cfg.METHOD.BACKBONE_CHECKPOINT, map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # ---- Learner (EMAT or CST) ----
        self.s_spatial_dim = math.isqrt(cfg.METHOD.SUPPORT_DIM - 1)
        self.learner = model_builder[cfg.METHOD.NAME](
            in_channels=self.nhead*self.nlayer, 
            t_s=cfg.METHOD.SUPPORT_DIM,
            fs_cs=self.fs_cs
        )
        
        # ---- Automatically defined by the MetricsCallback ----
        self.evaluator = None

    # ----------------------------------------------------------------------------------
    # PyTorch Lightning hooks
    # ----------------------------------------------------------------------------------
    
    def on_train_epoch_start(self) -> None:
        """Ensures the learner is in training mode while keeping the frozen backbone 
        in evaluation mode.
        """
        
        self.train()
        self.backbone.eval() 

    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run a single training step and log the loss.

        Parameters
        ----------
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader.

        Returns
        -------
        torch.Tensor
            Loss tensor.
        """
        
        return self.shared_step(batch)

    def on_train_epoch_end(self) -> None:
        """Log all metrics of the training epoch."""
        
        self.shared_epoch_end()

    def validation_step(self, batch: Dict[str, Any]) -> None:
        """Run a single validation step and log the loss.

        Parameters
        ----------
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader.
        """
        
        self.shared_step(batch)

    def on_validation_epoch_end(self) -> None:
        """Log all metrics of the validation epoch."""
        
        self.shared_epoch_end()

    def test_step(self, batch: Dict[str, Any]) -> None:
        """Run a single test step.

        Parameters
        ----------
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader.
        """
        
        pred_labels, pred_mask = self.predict_labels_and_mask(batch)
        
        if batch["task_augmented"]:
            self.augmented_tasks += 1
        
        if self.fs_cs:
            self.evaluator["test"].update_clf(pred_labels, batch)
        self.evaluator["test"].update_seg(pred_mask, batch)

    def on_test_epoch_end(self) -> None:
        """Aggregate test metrics, and optionally save them to a JSON file."""
        
        log_dict = {
            "method": self.method,
            "dataset": self.dataset,
            "setting": self.setting,
            "aumented_tasks": self.augmented_tasks,
            "fold": self.fold,
            "config": f"{self.way}-way {self.shot}-shot",
            "fs-cs": f"{self.fs_cs}",
            "empty-masks": f"{self.empty_masks}",
            "object-size": (
                "all objects" if self.object_size_split is None 
                else f"{self.object_size_split}%"
            ),
        }
        if self.fs_cs:
            log_dict["test/er"] = self.evaluator["test"].compute_avg_acc()
        log_dict["test/miou"] = self.evaluator["test"].compute_miou()
        
        print("\n")
        for k, v in log_dict.items():
            print(f"{k.ljust(20)}: {v}")
            
        if self.results_path is not None:
            results_file = Path(self.results_path)
            results_file.mkdir(parents=True, exist_ok=True)
            info = [
                self.method,
                self.dataset,
                self.setting,
                f"fold-{self.fold}",
                f"{self.way}w-{self.shot}s"
            ]
            if not self.fs_cs:
                info.append("only-seg")
            if not self.empty_masks:
                info.append("no-empty-masks")
            if self.object_size_split is not None:
                info.append(f"interval-{self.object_size_split}")
            results_file = results_file / ("_".join(info) + ".json")
            results_file.write_text(json.dumps(log_dict, indent=2, ensure_ascii=False))

    def configure_optimizers(self) -> torch.optim.Adam:
        """Build the optimizer over learner parameters only.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance configured with "self.lr".
        """
        return torch.optim.Adam([{"params": self.learner.parameters(), "lr": self.lr}])

    # ----------------------------------------------------------------------------------
    # Forward/inference helpers
    # ----------------------------------------------------------------------------------
    
    def forward(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Compute the correlation tokens for the N-way 1-shot tasks in the batch, and 
        process them with EMAT or CST to produce the multi-label classification vectors 
        and a multi-class segmentation masks.
        
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader:
            * "s_imgs": torch.Tensor - support images of shape "(B, Way, 3, H, W)" where 
            B is the batch size, and H and W correspond to the IMG_SIZE specified in the 
            configuration file.
            * "s_masks": torch.Tensor - support masks of shape "(B, Way, H, W)".
            * "q_img": torch.Tensor - query image of shape "(B, 3, H, W)".
            * "q_img_size": torch.Tensor - original size (H and W) of query image of 
            shape "(B, 2)".
        
        Returns
        -------
        Tuple[Optional[torch.Tensor], torch.Tensor]
            "(output_labels, output_masks)": output_labels is None if "self.fs_cs" is 
            False; otherwise, it is a tensor of shape "(B, Way, 2)". output_masks is a 
            tensor of shape "(B, Way, 2, H, W)" where H and W are the spatial dimensions 
            of the query image.
        """
        
        way = batch["s_imgs"].size(1)
        s_imgs = rearrange(batch["s_imgs"].squeeze(2), "b n c h w -> (b n) c h w")
        s_masks = rearrange(batch["s_masks"].squeeze(2), "b n h w -> (b n) h w")
        q_imgs = batch["q_img"]

        corr_tokens = self.feature_extraction(s_imgs, q_imgs, way)
        output_labels, output_masks = self.learner(corr_tokens, s_masks)

        if self.fs_cs:
            output_labels = output_labels.view(-1, way, 2)
        output_masks = self.upsample_masks(output_masks, batch)
        output_masks = output_masks.view(-1, way, *output_masks.shape[1:])
        return output_labels, output_masks
    
    @torch.no_grad()
    def feature_extraction(
        self, 
        s_imgs: torch.Tensor, 
        q_imgs: torch.Tensor, 
        way: int
    ) -> torch.Tensor:
        """Compute correlation tokens between support and query images.

        s_imgs : torch.Tensor
            Support images of shape "(B·Way, 3, H, W)" where B is the batch size, and H 
            and W correspond to the IMG_SIZE specified in the configuration file.
        q_imgs : torch.Tensor
            Query images of shape "(B, 3, H, W)".
        way : int
            Number of ways of the current task.
        
        Returns
        -------
        torch.Tensor
            Correlation tokens of shape "(B·Way, E, T_q, T_s)" where E is the embedding 
            dimension, T_q = h·w, and T_s = h'·w'+1.
        """
        
        s_tokens = self.backbone.get_intermediate_layers(s_imgs, n=self.nlayer)
        q_tokens = self.backbone.get_intermediate_layers(q_imgs, n=self.nlayer)

        s_tokens = torch.stack(s_tokens, dim=1)
        q_tokens = torch.stack(q_tokens, dim=1).repeat_interleave(way, dim=0)
        
        B, L, T, C = s_tokens.shape 
        s_tokens = s_tokens.reshape(B * L, T, C)
        q_tokens = q_tokens.reshape(B * L, T, C)

        s_img_tokens = s_tokens[:, 1:, :] 
        q_img_tokens = q_tokens[:, 1:, :] 
        
        s_img_tokens = rearrange(
            F.interpolate(
                rearrange(s_img_tokens, "b (h w) d -> b d h w", h=self.spatial_dim), 
                self.s_spatial_dim, 
                mode="bilinear", 
                align_corners=True
            ), 
            "b (n c) h w -> b n (h w) c", 
            n=self.nhead
        )
        q_img_tokens = rearrange(q_img_tokens, "b p (n c) -> b n p c", n=self.nhead)
        
        if self.fs_cs:
            s_cls_token = s_tokens[:, 0, :]
            s_cls_token = rearrange(s_cls_token, "b (n c) -> b n 1 c", n=self.nhead)
            s_img_tokens = torch.cat([s_cls_token, s_img_tokens], dim=2)

        s_img_tokens = F.normalize(s_img_tokens, p=2, dim=-1)
        q_img_tokens = F.normalize(q_img_tokens, p=2, dim=-1)

        corr_tokens = rearrange(
            torch.einsum("b n q c, b n s c -> b n q s", q_img_tokens, s_img_tokens), 
            "(b l) n q s -> b (n l) q s", 
            b=B
        )
        return corr_tokens
    
    def upsample_masks(self, masks: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Upsample the predicted masks to the size of the query image.

        masks : torch.Tensor
            Predicted masks of shape "(B·Way, 2, 100, 100)" where B is the batch size.
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader:
            * "q_img": torch.Tensor - query image of shape "(B, 3, H, W)".
            * "q_img_size": torch.Tensor - original size (H and W) of query image of 
            shape "(B, 2)".
        
        Returns
        -------
        torch.Tensor
            Upsampled masks of shape "(B·Way, 2, H, W)" where H and W are the spatial 
            dimensions of the query image.
        """
        
        if self.training:
            h, w = batch["q_img"].shape[-2:] 
        else:
            w, h = batch["q_img_size"][0]
        return F.interpolate(masks, (h,w), mode="bilinear", align_corners=True)
    
    # ----------------------------------------------------------------------------------
    # Training/validation shared logic
    # ----------------------------------------------------------------------------------
    
    def shared_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Run a single training or validation step.

        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader:
            * "s_imgs": torch.Tensor - support images of shape "(B, Way, 3, H, W)" where 
            B is the batch size, and H and W correspond to the IMG_SIZE specified in the 
            configuration file.
            * "s_masks": torch.Tensor - support masks of shape "(B, Way, H, W)".
            * "q_img": torch.Tensor - query image of shape "(B, 3, H, W)".
            * "q_img_size": torch.Tensor - original size (H and W) of query image of 
            shape "(B, 2)".
            * "q_classes": torch.Tensor - ground-truth labels of shape "(B, Way)".
            * "q_mask": torch.Tensor - ground-truth masks of shape "(B, H, W)".
        
        Returns
        -------
        torch.Tensor
            Loss of the step.
        """
        
        split = "trn" if self.training else "val"
        output_labels, output_masks = self.forward(batch)
        
        loss = self.compute_loss(
            output_labels, output_masks, batch["q_classes"], batch["q_mask"]
        )
        self.log(
            f"{split}/loss", 
            loss, 
            on_step=False, 
            on_epoch=True, 
            logger=True, 
            sync_dist=True
        )
            
        with torch.no_grad():
            if self.fs_cs:
                pred_labels = self.predict_labels(output_labels)
                self.evaluator[split].update_clf(pred_labels, batch)
            
            pred_masks = self.predict_masks(output_masks)
            self.evaluator[split].update_seg(pred_masks, batch)
            self.evaluator[split].update_loss(loss.item())
        return loss

    def shared_epoch_end(self) -> None:
        """Log all metrics of a training or validation epoch."""
        
        split = "trn" if self.training else "val"
        metrics = {f"{split}/loss": self.evaluator[split].compute_avg_loss()}
        if self.fs_cs:
            metrics[f"{split}/er"] = self.evaluator[split].compute_avg_acc() 
        metrics[f"{split}/miou"] = self.evaluator[split].compute_miou()

        space = "\n\n" if split == "val" else "\n"
        info = [f"{space}{split}/epoch: {self.current_epoch:>3}"]
        for k, v in metrics.items():
            self.log(
                k, 
                v, 
                on_step=False, 
                on_epoch=True, 
                logger=True, 
                sync_dist=True
            )
            info.append(f"{k}: {v:.3f}")
        print(" | ".join(info))
      
    # ----------------------------------------------------------------------------------
    # Loss and predictions
    # ----------------------------------------------------------------------------------

    def compute_loss(
        self, 
        output_labels: Optional[torch.Tensor], 
        output_masks: torch.Tensor, 
        gt_labels: torch.Tensor, 
        gt_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute the adequate loss for Few-Shot Classification and Segmentation or 
        Few-Shot Segmentation, but it only supports 1-way tasks

        output_labels : Optional[torch.Tensor]
            Predicted labels is None if "self.fs_cs" is False; otherwise, it is a tensor 
            of shape "(B, Way, 2)" where B is the batch size.
        output_masks : torch.Tensor
            Predicted masks of shape "(B, Way, 2, H, W)" where H and W are the spatial 
            dimensions of the query image.
        gt_labels : torch.Tensor
            Ground-truth labels of shape "(B, Way)".
        gt_masks : torch.Tensor
            Ground-truth masks of shape "(B, H, W)".
        
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        
        cls_loss = torch.tensor(0.0, device=output_masks.device)
        if self.fs_cs:
            logit_labels = torch.log_softmax(output_labels, dim=2).squeeze(1)
            cls_loss = F.nll_loss(logit_labels, gt_labels.long().squeeze(-1))
            
        logit_masks = torch.log_softmax(output_masks, dim=2).squeeze(1)
        seg_loss = F.nll_loss(logit_masks, gt_masks.long())
        return cls_loss * self.loss_balance + seg_loss
    
    @torch.no_grad()
    def predict_labels(self, output_labels: torch.Tensor) -> torch.Tensor:
        """Apply thresholding to the predicted labels.

        output_labels : torch.Tensor
            Predicted labels of shape "(B, Way, 2)" where B is the batch size.
        
        Returns
        -------
        torch.Tensor
            Thresholded labels of shape "(B, Way)".
        """
        
        labels_probs = torch.softmax(output_labels, dim=2)
        return labels_probs[..., 1] > self.delta

    @torch.no_grad()
    def predict_masks(self, output_masks: torch.Tensor) -> torch.Tensor:
        """Apply thresholding to the predicted masks.

        output_masks : torch.Tensor
            Predicted masks of shape "(B, Way, 2, H, W)" where B is the batch size, and 
            H and W are the spatial dimensions of the query image.
        
        Returns
        -------
        torch.Tensor
            Thresholded masks of shape "(B, H, W)".
        """
        
        masks_probs = torch.softmax(output_masks, dim=2)
        max_fg_val, pred_masks = masks_probs[:, :, 1].max(dim=1)
        pred_masks = pred_masks + 1 
        pred_masks[max_fg_val < self.delta] = 0  
        return pred_masks
    
    @torch.no_grad()
    def predict_labels_and_mask(
        self, 
        batch: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Process a N-way K-shot task as K N-way 1-shot tasks.
        
        batch : Dict[str, Any]
            Batch dictionary produced by the few-shot dataloader:
            * "s_imgs": torch.Tensor - support images of shape "(B, Way, Shot, 3, H, W)" 
            where B is the batch size, and H and W correspond to the IMG_SIZE specified 
            in the configuration file.
            * "s_masks": torch.Tensor - support masks of shape "(B, Way, Shot, H, W)".
        
        Returns
        -------
        Tuple[Optional[torch.Tensor], torch.Tensor]
            "(pred_labels, pred_mask)": pred_labels is None if "self.fs_cs" is 
            False; otherwise, it is a tensor of shape "(B, Way)". output_masks is a 
            tensor of shape "(B, H, W)" where H and W are the spatial dimensions of the 
            query image.
        """
        
        s_imgs = batch["s_imgs"].clone()
        s_masks = batch["s_masks"].clone()
        
        output_labels = 0
        output_mask = 0
        for i in range(s_imgs.size(2)):
            batch["s_imgs"] = s_imgs[:, :, i]
            batch["s_masks"] = s_masks[:, :, i]
            output_labels_i, output_masks_i = self.forward(batch)
            if self.fs_cs:
                output_labels += torch.softmax(output_labels_i, dim=2)
            output_mask += torch.softmax(output_masks_i, dim=2)
        
        pred_labels = self.predict_labels(output_labels) if self.fs_cs else None
        pred_mask = self.predict_masks(output_mask)

        return pred_labels, pred_mask
