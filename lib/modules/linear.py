import torch.nn as nn
import torch
from conv1d import Conv1D


class MLP(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.c_fc = Conv1D(d_model, dim_feedforward)
        self.c_proj = Conv1D(dim_feedforward, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)