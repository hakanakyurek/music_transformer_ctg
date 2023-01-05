import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from lib.utilities.constants import *
from lib.utilities.device import get_device

from ..modules.positional_encoding import PositionalEncoding
from ..modules.encoder_rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR
from ..modules.decoder_rpr import TransformerDecoderRPR, TransformerDecoderLayerRPR

from .music_transformer_base import MusicTransformerBase


class MusicTransformerEncoderDecoder(MusicTransformerBase):

    def __init__(self, loss_fn, acc_metric, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False, lr=1.0):
        super(MusicTransformerEncoderDecoder, self).__init__(acc_metric)
        self.nlayers      = n_layers
        self.nhead        = num_heads
        self.d_model      = d_model
        self.d_ff         = dim_feedforward
        self.dropout      = dropout
        self.max_seq      = max_sequence
        self.rpr          = rpr
        self.loss_fn      = loss_fn
        self.lr           = lr

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff
            )
        # RPR Transformer
        else:
            # Init Encoder
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            # Init Decoder
            decoder_layer = TransformerDecoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            decoder = TransformerDecoderRPR(decoder_layer, self.nlayers)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=decoder, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, tgt, mask=True):
        """
    
        Author: Damon Gwinn
    
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
    
        """

        if(mask is True):
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = self.create_masks(x, tgt)
        else:
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = None, None, None, None

        x = self.embedding(x)
        tgt = self.embedding(tgt)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)
        tgt = tgt.permute(1, 0 ,2)

        x = self.positional_encoding(x)
        tgt = self.positional_encoding(tgt)

        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=tgt, 
                                 src_mask=src_mask, 
                                 tgt_mask=tgt_mask, 
                                 src_key_padding_mask=src_pad_mask,
                                 tgt_key_padding_mask=tgt_pad_mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # source: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float(-1e9)) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def create_masks(self, src, tgt):
        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[1])
        tgt_mask = self.get_tgt_mask(tgt.shape[1])
        src_pad_mask = self.create_pad_mask(src, TOKEN_PAD)
        tgt_pad_mask = self.create_pad_mask(tgt, TOKEN_PAD)

        return src_mask.to(get_device()), \
               tgt_mask.to(get_device()), \
               src_pad_mask.to(get_device()), \
               tgt_pad_mask.to(get_device())

    def step(self, batch, acc_metric, pp_metric):
        
        x, tgt_input, tgt_output = batch
        # tgt is shifted to the right by 1
        y = self.forward(x, tgt_input)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt_output = tgt_output.flatten()

        loss = self.loss_fn.forward(y, tgt_output)

        self.metric_update(acc_metric, pp_metric, y, tgt_output)

        return loss, y
