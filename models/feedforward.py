import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP with residual connection and group normalization.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    groups : int, default=4
        Number of groups for group normalization.
    size : int, default=2
        Expansion factor for the hidden layer, i.e. "hidden = out//size".
    """
    
    def __init__(self, in_channels: int, groups: int = 4, size: int = 2) -> None:
        super().__init__()
        hidden_channels = in_channels // size
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.out_norm = nn.GroupNorm(groups, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual addition.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape "(B·Way, E, T_q, T_d)" where B is the batch size, E 
            the embedding dimension, T_q = h·w, and T_d = h''·w''+1.

        Returns
        -------
        torch.Tensor
            Output tensor of shape "(B·Way, E, T_q, T_d)".
        """
        
        out = self.ff(x)
        out.add_(x)
        return self.out_norm(out)
