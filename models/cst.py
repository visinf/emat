from typing import Any, Tuple

import torch.nn as nn
from torch import Tensor

from models.attention import MaskedAttention
from models.base_model import BaseModel
from models.feedforward import FeedForward


class MaskedAttentionLayer(nn.Module):
    """Transformer layer consisting of masked attention followed by a feed-forward MLP.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    out_channels : int
        Embedding dimension of the output tokens.
    pooling_window : int
        Window size used for downscaling the support spatial dimensions inside the 
        masked attention mechanism.
    """
    
    def __init__(self, in_channels: int, out_channels: int, pooling_window: int) -> None:
        super().__init__()
        self.attn = MaskedAttention(in_channels, out_channels, pooling_window)
        self.ff = FeedForward(out_channels, groups=4)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply masked attention followed by a residual feed-forward block.

        Parameters
        ----------
        input : Tuple[torch.Tensor, torch.Tensor]
            "(tokens, s_masks)" with tokens of shape "(B·Way, in_channels, T_q, T_s)" 
            where B is the batch size, T_q = h·w, and T_s = h'·w'+1. s_masks of shape 
            "(B·Way, H, W)" where H and W correspond to the IMG_SIZE specified in the 
            configuration file.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            "(tokens, s_masks)" where tokens are the processed tokens of shape 
            "(B·Way, out_channels, T_q, T_d)" with T_d = h''·w''+1, and s_masks are the 
            support masks provided as input.
        """
        
        tokens, s_masks = input
        tokens = self.attn((tokens, s_masks))
        tokens = self.ff(tokens)
        return tokens, s_masks
    
    
class CST(BaseModel):
    """Classification and Segmentation Transformer from Kang et. al.

    Parameters
    ----------
    in_channels : int
        Embedding dimension of the correlation tokens.
    t_s : int
        Support dimension of the first attention layer. Only "145" and "325" are 
        supported.
    **kwargs : Any
        Present for compatibility across models.
    """
    
    def __init__(self, in_channels: int, t_s: int, **kwargs: Any) -> None:
        hidden_channels = 32
        out_channels = 128
        super().__init__(out_channels, fs_cs=True)
        
        if t_s == 145:
            self.corrtransformer = nn.Sequential(
                MaskedAttentionLayer(in_channels, hidden_channels, pooling_window=4),
                MaskedAttentionLayer(hidden_channels, out_channels, pooling_window=3)
            )
        elif t_s == 325:
            self.corrtransformer = nn.Sequential(
                MaskedAttentionLayer(in_channels, hidden_channels, pooling_window=3),
                MaskedAttentionLayer(hidden_channels, out_channels, pooling_window=6)
            )
        else:
            raise ValueError("'t_s' must be either 145 or 325.")
