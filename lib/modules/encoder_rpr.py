import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from .multihead_attention import MultiheadAttentionRPR

# TransformerEncoderRPR
class TransformerEncoderRPR(Module):
    """
    ----------
    Author: Pytorch
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoder

    No modification. Copied here to ensure continued compatibility with other edits.
    ----------
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderRPR, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

# TransformerEncoderLayerRPR
class TransformerEncoderLayerRPR(Module):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

    Modification to create and call custom MultiheadAttentionRPR
    ----------
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerEncoderLayerRPR, self).__init__()
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.silu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
