# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        Arguments:
            x: Tensor, (batch_size, position)

        Output:
            x: Tensor, (batch_size, d_model) 
        """
        x = self.pe[x] 
        return x


