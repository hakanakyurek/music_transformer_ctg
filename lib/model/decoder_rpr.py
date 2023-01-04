import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.modules.transformer import _get_clones
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from .multihead_attention import MultiheadAttentionRPR


class TransformerDecoderRPR(Module):
    def __init__(self, decoder_layer, num_layers, norm) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

        output = tgt
        
        return output


class TransformerDecoderLayerRPR(Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, er_len=None):
        super(TransformerDecoderLayerRPR, self).__init__()
        self.self_attn = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        
        self.enc_dec_attention = MultiheadAttentionRPR(d_model, nhead, dropout=dropout, er_len=er_len)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout3 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm3 = LayerNorm(d_model)
        self.dropout4 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # 1. compute self attention
        _x = tgt
        x = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask,
                           key_padding_mask=tgt_key_padding_mask)[0]
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if memory is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(query=x, key=memory, value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.linear1(x)
        x = self.dropout3(F.silu(x))
        x = self.linear2(x)
        
        # 6. add and norm
        x = self.dropout4(x)
        x = self.norm3(x + _x)
        return x

