import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Base model for Few-shot Classification and Segmentation (FS-CS) or Few-Shot 
    Segmentation (FS-S).

    Parameters
    ----------
    out_channels : int
        Number of channels of the tokens produced by the correlation transformer (CST or 
        EMAT).
    fs_cs : bool, default=True
        Whether the incoming correlation tokens corresponds to FS-CS, i.e., whether it 
        contains a support class token.
    """
    
    def __init__(self, out_channels: int, fs_cs: bool = True) -> None:
        super().__init__()
        
        self.corrtransformer = None # to be set by the caller class
        self.fs_cs = fs_cs

        modules = []
        for _ in range(2):
            modules.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
            )
            modules.append(nn.GroupNorm(4, out_channels))
            modules.append(nn.ReLU(inplace=True))
        self.linear = nn.Sequential(*modules)
        
        # ---- Specialized heads ----
        if self.fs_cs:
            self.decoder1_cls = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels//2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
            )

            self.decoder2_cls = nn.Sequential(
                nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels//2, 2, kernel_size=1, bias=True)
            )
        
        self.decoder1_seg = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels//2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2_seg = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, 2, 3, padding=1, bias=True)
        )

    def forward(
        self, 
        corr_tokens: torch.Tensor, 
        s_masks: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run the correlation transformer and process its output using the specialized
        heads for classification and segmentation.

        Parameters
        ----------
        corr_tokens : torch.Tensor
            Correlation tokens of shape "(B·Way, E, T_q, T_s)" where B is the batch 
            size, E the embedding dimension, T_q = h·w, and T_s = h'·w'+1 if self.fs_cs 
            is True, otherwise T_s = h'·w'.
        s_masks : torch.Tensor
            Support masks of shape "(B·Way, H, W)" where H and W correspond to the 
            IMG_SIZE specified in the configuration file.

        Returns
        -------
        Tuple[Optional[torch.Tensor], torch.Tensor]
            "(output_labels, output_masks)" where output_labels is "None" when
            self.fs_cs is False; otherwise, it is a tensor of shape "(Ways, 2)". 
            output_masks has shape "(B·Way, 2, 2·T_q, 2·T_q)" where T_q = h·w.
        """
        
        tokens = self.corrtransformer((corr_tokens, s_masks))[0]
        tokens = self.linear(tokens)
        
        B, C, T_q, _ = tokens.shape
        q_spatial_dim = math.isqrt(T_q)

        if self.fs_cs:
            cls_token, seg_token = tokens[..., 0], tokens[..., 1]
            
            # ---- Classification ----
            cls_token = cls_token.view(B, C, q_spatial_dim, q_spatial_dim)
            output_labels = self.decoder1_cls(cls_token)
            output_labels = self.decoder2_cls(
                F.avg_pool2d(output_labels, kernel_size=3, stride=2, padding=1) 
            )
            output_labels = F.adaptive_avg_pool2d(output_labels, 1).squeeze()
        else:
            seg_token = tokens[..., 0]
            output_labels = None
            
        # ---- Segmentation ----
        seg_token = seg_token.view(B, C, q_spatial_dim, q_spatial_dim)
        output_masks = F.interpolate(
            self.decoder1_seg(seg_token), 
            q_spatial_dim*2, 
            mode="bilinear", 
            align_corners=True
        )
        output_masks = self.decoder2_seg(output_masks)

        return output_labels, output_masks
