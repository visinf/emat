import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.pooling import CSTPooling, LearnableDownscaling


class MaskedAttention(nn.Module):
    """Masked attention layer with residual connection.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    out_channels : int
        Embedding dimension of the output tokens.
    pooling_window : int
        Window size used for downscaling the support spatial dimensions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, pooling_window: int) -> None:
        super().__init__()
        self.heads = 8
        groups = 4
        hidden_channels = out_channels // 2

        pool_params = (1, pooling_window, pooling_window)

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            CSTPooling(pool_params),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        ) 
        
        self.qkv = nn.Conv2d(in_channels, hidden_channels * 3, kernel_size=1, bias=True)
        self.avgpool = CSTPooling(pool_params)
        
        self.agg = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply masked attention with residual connection.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            "(corr_tokens, s_masks)" with corr_tokens of shape "(B·Way, in_channels, 
            T_q, T_s)" where B is the batch size, T_q = h·w, and T_s = h'·w'+1. s_masks 
            of shape "(B·Way, H, W)" where H and W correspond to the IMG_SIZE specified 
            in the configuration file.

        Returns
        -------
        torch.Tensor
            Processed tokens of shape "(B·Way, out_channels, T_q, T_d)" with 
            T_d = h''·w''+1.
        """
        
        corr_tokens, s_masks = input 
        residual = self.short_cut(corr_tokens)

        qkv = self.qkv(corr_tokens) 
        qkv = rearrange(qkv, 'b (x g e) q s -> x b g e q s', x=3, g=self.heads) 
        q, k, v = qkv
        
        q = rearrange(q, 'b g e q s -> b (g e) q s')
        q = self.avgpool(q)
        q = rearrange(q, 'b (g e) q d -> b g e q d', g=self.heads)
        
        out = self.masked_attn(q, k, v, s_masks)        
        out = self.agg(out)

        return self.out_norm(out + residual)

    def masked_attn(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        s_masks: torch.Tensor
    ) -> torch.Tensor:
        """Apply masked attention with residual connection.

        Parameters
        ----------
        q : torch.Tensor
            Query matrix of shape "(B·Way, G, E, T_q, T_d)" where B is the batch size, G 
            is the number of heads, E the head dimension, T_q = h·w, and T_d = h''·w''+1.
        k : torch.Tensor
            Key matrix of shape "(B·Way, G, E, T_q, T_s)" where T_s = h'·w'+1.
        v : torch.Tensor
            Value matrix of shape "(B·Way, G, E, T_q, T_s)".
        s_masks : torch.Tensor
            Support masks of shape "(B·Way, H, W)" where H and W correspond to the 
            IMG_SIZE specified in the configuration file.
            
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape "(B·Way, out_channels, T_q, T_d)".
        """
        
        attn = torch.einsum('b g e q d, b g e q s -> b g q d s', q, k)
        attn = F.softmax(self.mask(attn, s_masks), dim=-1)
        attn = torch.einsum('b g q d s, b g e q s -> b g e q d', attn, v)
        return rearrange(attn, 'b g e q d -> b (g e) q d')
    
    def mask(self, attn: torch.Tensor, s_masks: torch.Tensor) -> torch.Tensor:
        """Mask the attention matrix based on the support mask.

        Parameters
        ----------
        attn : torch.Tensor
            Attention matrix of shape "(B·Way, G, T_q, T_d, T_s)" where B is the batch 
            size, G is the number of heads, T_q = h·w, T_d = h''·w''+1, and T_s = h'·w'
            +1.
        s_masks : torch.Tensor
            Support masks of shape "(B·Way, H, W)" where H and W correspond to the 
            IMG_SIZE specified in the configuration file.
            
        Returns
        -------
        torch.Tensor
            Masked attention matrix of shape "(B·Way, G, T_q, T_d, T_s)".
        """
        
        s_spatial_dim = math.isqrt(attn.size(-1) - 1)
        masks = s_masks.clone().clamp_max_(1).float().unsqueeze(1)
        masks = F.interpolate(masks, s_spatial_dim, mode='bilinear', align_corners=True)
        masks = rearrange(masks, 'b 1 h w -> b 1 1 1 (h w)')
        
        attn_seg = attn[..., 1:].masked_fill_(masks == 0, -1e9)
        return torch.cat([attn[..., :1], attn_seg], dim=-1)
            
    
class EfficientMaskedAttention(nn.Module):
    """Efficient masked attention layer with residual connection.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    out_channels : int
        Embedding dimension of the output tokens.
    pooling_window : int
        Window size used for downscaling the support spatial dimensions.
    fs_cs : bool, default=True
        Whether the incoming correlation tokens corresponds to Few-shot Classification 
        and Segmentation (FS-CS), i.e., whether it contains a support class token.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pooling_window: int, 
        fs_cs: bool = True
    ) -> None:
        super().__init__()
        self.fs_cs = fs_cs
        self.heads = 8
        groups = 4
        hidden_channels = out_channels // 2

        conv_params = (1, 2, 2) 
        pool_params = None if pooling_window==0 else (1, pooling_window, pooling_window)
        
        self.short_cut = nn.Sequential(
            LearnableDownscaling(
                in_channels, 
                out_channels, 
                conv_params, 
                pool_params, 
                bias=False,
                fs_cs=self.fs_cs
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )

        self.q = LearnableDownscaling(
            in_channels, 
            hidden_channels, 
            conv_params, 
            pool_params, 
            bias=True,
            fs_cs=self.fs_cs
        )
        self.kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=True)

        self.agg = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply efficient masked attention with residual connection.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            "(corr_tokens, s_masks)" with corr_tokens of shape "(B·Way, in_channels, 
            T_q, T_s)" where B is the batch size, T_q = h·w, and T_s = h'·w'+1. s_masks 
            of shape "(B·Way, H, W)" where H and W correspond to the IMG_SIZE specified 
            in the configuration file.

        Returns
        -------
        torch.Tensor
            Processed tokens of shape "(B·Way, out_channels, T_q, T_d)" with 
            T_d = h''·w''+1.
        """
        
        corr_tokens, s_masks = input 
        residual = self.short_cut(corr_tokens)

        q = self.q(corr_tokens)
        q = rearrange(q, 'b (g e) q d -> b g e q d', g=self.heads)
        
        kv = self.kv(corr_tokens) 
        kv = rearrange(kv, 'b (x g e) q s -> x b g e q s', x=2, g=self.heads)
        k, v = kv
        
        out = self.efficient_masked_attn(q, k, v, s_masks)        
        out = self.agg(out)

        return self.out_norm(out + residual)

    def efficient_masked_attn(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        s_masks: torch.Tensor
    ) -> torch.Tensor:
        """Apply masked attention with residual connection.

        Parameters
        ----------
        q : torch.Tensor
            Query matrix of shape "(B·Way, G, E, T_q, T_d)" where B is the batch size, G 
            is the number of heads, E the head dimension, T_q = h·w, and T_d = h''·w''+1.
        k : torch.Tensor
            Key matrix of shape "(B·Way, G, E, T_q, T_s)" where T_s = h'·w'+1.
        v : torch.Tensor
            Value matrix of shape "(B·Way, G, E, T_q, T_s)".
        s_masks : torch.Tensor
            Support masks of shape "(B·Way, H, W)" where H and W correspond to the 
            IMG_SIZE specified in the configuration file.
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape "(B·Way, out_channels, T_q, T_d)".
        """
        
        B, _, _, _, T_s = k.shape
        
        offset = 1 if self.fs_cs else 0
        s_spatial_dim = math.isqrt(T_s - offset)
        
        masks = s_masks.clone().clamp_max_(1).float().unsqueeze(1)
        masks = F.interpolate(masks, s_spatial_dim, mode='bilinear', align_corners=True)
        masks = rearrange(masks, 'b 1 h w -> b (h w)').bool()
        
        out = []
        for i in range(B):
            k_i, v_i = self.mask(k[i], v[i], masks[i])
            out_i = torch.einsum('g e q d, g e q m -> g q d m', q[i], k_i)
            out_i = F.softmax(out_i, dim=-1)
            out_i = torch.einsum('g q d m, g e q m -> g e q d', out_i, v_i)
            out.append(out_i)
        out = torch.stack(out)
        return rearrange(out, 'b g e q d -> b (g e) q d')
            
    def mask(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        bool_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply masked attention with residual connection.

        Parameters
        ----------
        k : torch.Tensor
            Key matrix of shape "(G, E, T_q, T_s)" where G is the number of heads, E the 
            head dimension, T_q = h·w, and T_s = h'·w'+1.
        v : torch.Tensor
            Value matrix of shape "(G, E, T_q, T_s)".
        bool_mask : torch.Tensor
            Boolean mask of shape "(h'·w')".
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            "(k, v)" masked Key and Value matrices of shape "(G, E, T_q, M)" where M is 
            the number of non-removed tokens, which varies for each image
        """
        
        if self.fs_cs: 
            k = torch.cat([k[..., :1], k[..., 1:][..., bool_mask]], dim=-1)
            v = torch.cat([v[..., :1], v[..., 1:][..., bool_mask]], dim=-1)
        else:
            k = k[..., bool_mask]
            v = v[..., bool_mask]
        return k, v
