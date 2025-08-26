import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


class CSTPooling(nn.Module):
    """Average-pooling-based strategy for downscaling the support spatial dimensions.

    Parameters
    ----------
    pool_params : Tuple[int, int, int]
        Kernel and stride for the 3D average pooling applied to image tokens.
    """
    
    def __init__(self, pool_params: Tuple[int, int, int]) -> None:
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size=pool_params, stride=pool_params, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply average pool to image tokens while keeping the class token intact.

        Parameters
        ----------
        x : torch.Tensor
            Correlation tokens of shape "(B·Way, E, T_q, T_s)" where B is the batch 
            size, E the embedding dimension, T_q = h·w, and T_s = h'·w'+1.

        Returns
        -------
        torch.Tensor
            Tensor with support spatial dimensions (h' and w') reduced by "kernel_size".
        """
        
        s_spatial_dim = math.isqrt(x.size(-1) - 1)
        s_img_tokens = self.pool(
            rearrange(x[..., 1:], 'b e q (h w) -> b e q h w', h=s_spatial_dim)
        )
        s_img_tokens = rearrange(s_img_tokens, 'b e q h w -> b e q (h w)')
        return torch.cat([x[..., 0].unsqueeze(-1), s_img_tokens], dim=-1)
    
    
class LearnableDownscaling(nn.Module):
    """Learnable downscaling strategy for downscaling the support spatial dimensions.
    This combines strategy combines small convolutions with average pooling if needed.

    Parameters
    ----------
    in_channels : int
        Channel dimension of the incoming correlation tokens.
    hidden_channels : int
        Channel dimension after downscaling.
    conv_params : Tuple[int, int, int]
        Kernel and stride for the 3D convolution applied to image tokens.
    pool_params : None | Tuple[int, int, int], default=None
        If not "None", kernel and stride for the 3D average pooling applied to image 
        tokens after the convolution.
    bias : bool, default=True
        Whether to include bias terms in the convolution layers.
    fs_cs : bool, default=True
        Whether the incoming correlation tokens corresponds to Few-shot Classification 
        and Segmentation (FS-CS), i.e., whether it contains a support class token.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        conv_params: Tuple[int, int, int], 
        pool_params: Optional[Tuple[int, int, int]] = None, 
        bias: bool = True,
        fs_cs: bool = True
    ) -> None:
        super().__init__()
        self.fs_cs = fs_cs
        if self.fs_cs:
            self.q_cls = nn.Conv2d(
                in_channels, 
                hidden_channels, 
                kernel_size=1, 
                bias=bias
            )
        self.q_seg = nn.Conv3d(
            in_channels, 
            hidden_channels, 
            kernel_size=conv_params, 
            stride=conv_params, 
            padding=0, 
            bias=bias
        )
        self.pool = (
            None if pool_params is None 
            else nn.AvgPool3d(kernel_size=pool_params, stride=pool_params, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply our proposed learnable downscaling strategy to the incoming correlation 
        tokens.

        Parameters
        ----------
        x : torch.Tensor
            Correlation tokens of shape "(B·Way, E, T_q, T_s)" where B is the batch 
            size, E the embedding dimension, T_q = h·w, and T_s = h'·w'+1 if self.fs_cs 
            is True, otherwise T_s = h'·w'.

        Returns
        -------
        torch.Tensor
            Tensor with support spatial dimensions (h' and w') reduced by 3D convolution 
            (and optionally by 3D average pooling).
        """
        offset = 1 if self.fs_cs else 0
        s_spatial_dim = math.isqrt(x.size(-1) - offset)
        
        s_img_tokens = self.q_seg(
            rearrange(x[..., offset:], 'b e q (h w) -> b e q h w', h=s_spatial_dim)
        )
        if self.pool is not None:
            s_img_tokens = self.pool(s_img_tokens)
        s_img_tokens = rearrange(s_img_tokens, 'b e q h w -> b e q (h w)')
        
        if self.fs_cs:
            s_cls_token = self.q_cls(x[..., 0].unsqueeze(-1))  
            return torch.cat([s_cls_token, s_img_tokens], dim=-1)
        return s_img_tokens
