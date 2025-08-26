from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

from models.attention import EfficientMaskedAttention
from models.base_model import BaseModel
from models.feedforward import FeedForward


class EfficientMaskedAttentionLayer(nn.Module):
    """Transformer layer consisting of efficient masked attention followed by a 
    feed-forward MLP.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    out_channels : int
        Embedding dimension of the output tokens.
    pooling_window : int
        Window size used for downscaling the support spatial dimensions inside the 
        masked attention mechanism.
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
        self.attn = EfficientMaskedAttention(
            in_channels, 
            out_channels, 
            pooling_window,
            fs_cs
        )
        self.ff = FeedForward(out_channels, 4)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply efficient masked attention followed by a residual feed-forward block.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            "(tokens, s_masks)" with tokens of shape "(B·Way, in_channels, T_q, T_s)" 
            where B is the batch size, T_q = h·w, and T_s = h'·w'+1 if fs_cs is True, 
            otherwise T_s = h'·w'. s_masks of shape "(B·Way, H, W)" where H and W 
            correspond to the IMG_SIZE specified in the configuration file.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            "(tokens, s_masks)" where tokens are the processed tokens of shape 
            "(B·Way, out_channels, T_q, T_d)" with T_d = h''·w''+1 if fs_cs is True, 
            otherwise T_d = h'·w', and s_masks are the support masks provided as input.
        """
        
        tokens, s_masks = input
        tokens = self.attn((tokens, s_masks))
        tokens = self.ff(tokens)
        return tokens, s_masks


class EMAT(BaseModel):
    """Efficient Masked Attention Transformer.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    fs_cs : bool, default=True
        Whether the incoming correlation tokens corresponds to Few-shot Classification 
        and Segmentation (FS-CS), i.e., whether it contains a support class token.
    **kwargs : Any
        Present for compatibility across models.
    """
    
    def __init__(self, in_channels: int, fs_cs: bool = True, **kwargs: Any) -> None:
        hidden_channels = 64
        out_channels = 32
        super().__init__(out_channels, fs_cs=fs_cs)

        self.corrtransformer = nn.Sequential(
            EfficientMaskedAttentionLayer(
                in_channels, 
                hidden_channels, 
                pooling_window=0, 
                fs_cs=fs_cs
            ),
            EfficientMaskedAttentionLayer(
                hidden_channels, 
                out_channels, 
                pooling_window=5, 
                fs_cs=fs_cs
            )
        )
