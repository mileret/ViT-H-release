# Sinusoidal positional encoding, used here as a lookup table for diffusion
# timestep embeddings (pe[t] -> sinusoidal vector for integer step t).
# Copied from GEM-X (gem/network/base_arch/embeddings/pe.py).
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x, batch_first=False):
        if batch_first:
            pe = self.pe.transpose(0, 1)
            x = x + pe[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)
