'''
    Deep Learning module of crip.
    
    https://github.com/z0gSh1u/crip
'''

__all__ = ['TotalVariation']

import torch
import torch.nn as nn

from .lowdose import totalVariation


class TotalVariation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.from_numpy(totalVariation(x.numpy()))
