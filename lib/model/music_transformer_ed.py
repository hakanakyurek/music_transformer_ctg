import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from lib.utilities.constants import *
from lib.utilities.device import get_device
from lib.utilities.top_p_top_k import top_k_top_p_filtering

from ..modules.positional_encoding import PositionalEncoding
from ..modules.encoder_rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR
from ..modules.decoder_rpr import TransformerDecoderRPR, TransformerDecoderLayerRPR

from .music_transformer_base import MusicTransformerBase


class MusicTransformerEncoderDecoder(MusicTransformerBase):

    def __init__(self, loss_fn=None, acc_metric=None, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
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

    # generate
    def generate(self, primer=None, target_seq_length=1024, temperature=1.0, top_p=0.0, top_k=0):
        """
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        """

        assert (not self.training), "Cannot generate while in training mode"

        print(f"Generating sequence of max length: {target_seq_length}")

        prior = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        prior[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq[..., 0] = TOKEN_START
        gen_seq[..., 1 :num_primer + 1] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        # print("primer:",primer)
        # print(gen_seq.shape)
        # Here cur_i is the current token index
        cur_i = num_primer
        while(cur_i < target_seq_length):

            logits = self.forward(prior[..., :cur_i], gen_seq[..., :cur_i])
            
            token_probs = self.softmax(logits)[..., :TOKEN_END]
            token_probs = token_probs[:, cur_i-1, :]
            # next_token = torch.argmax(token_probs)
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            print(distrib)
            # print("next token:",next_token)
            gen_seq[:, cur_i] = next_token

            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                print(f"Model called end of sequence at: {cur_i}/{target_seq_length}")
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(f"{cur_i}/{target_seq_length}")

        return gen_seq[:, :cur_i]

    # source: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        # Padding is done in the dataset __getitem__
        return (matrix == pad_token)

    def create_masks(self, src, tgt):
        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[1])
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
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
        
        pp_metric.update(y, tgt_output)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt_output = tgt_output.flatten()

        loss = self.loss_fn.forward(y, tgt_output)

        acc_metric.update(y, tgt_output)

        return loss, y
