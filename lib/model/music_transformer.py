import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.optim.lr_scheduler import LambdaLR

from lib.utilities.constants import *
from lib.utilities.device import get_device
from lib.utilities.lr_scheduling import LrStepTracker, get_lr

from ..modules.positional_encoding import PositionalEncoding
from ..modules.encoder_rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR

import logging
import random

from lib.utilities.top_p_top_k import top_k_top_p_filtering
import pytorch_lightning as pl


# MusicTransformer
class MusicTransformer(pl.LightningModule):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, loss_fn, acc_metric, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False, lr=1.0):
        super(MusicTransformer, self).__init__()
        print(f"Creating the music transformer")
        self.dummy        = DummyDecoder()

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
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

        self.train_acc = acc_metric()
        self.val_acc = acc_metric()
        self.test_acc = acc_metric()

    # forward
    def forward(self, x, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        x = self.embedding(x)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1,0,2)

        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

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
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """

        assert (not self.training), "Cannot generate while in training mode"

        logging.info(f"Generating sequence of max length: {target_seq_length}")

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # logging.info("primer:",primer)
        # print(gen_seq.shape)
        # Here cur_i is the current token index
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            logits = self.forward(gen_seq[..., :cur_i])
            
            logits = logits[:, cur_i-1, :] / temperature

            if top_p != 0.0 and top_k != 0:
                logits = top_k_top_p_filtering(logits, top_k, top_p)

            token_probs = self.softmax(logits)[..., :TOKEN_END]

            
            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            # logging.info("next token:",next_token)
            gen_seq[:, cur_i] = next_token


            # Let the transformer decide to end if it wants to
            if(next_token == TOKEN_END):
                logging.info(f"Model called end of sequence at: {cur_i}/{target_seq_length}")
                break

            cur_i += 1
            if(cur_i % 50 == 0):
                logging.info(f"{cur_i}/{target_seq_length}")

        return gen_seq[:, :cur_i]

    def step(self, batch, acc_metric):
        
        x, tgt = batch

        y = self.forward(x)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        loss = self.loss_fn.forward(y, tgt)

        acc_metric.update(y, tgt)

        return loss, y

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.train_acc)

        self.log('training loss', loss)

        return loss

    def training_epoch_end(self, outs):
        accuracy = self.train_acc.compute()
        self.log('train accuracy', accuracy)

    def validation_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.val_acc)
        self.log('validation loss', loss)

        return loss
    
    def validation_epoch_end(self, outs):
        accuracy = self.val_acc.compute()
        self.log('validation accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        loss, y = self.step(batch, self.test_acc)
        return loss, y

    def test_epoch_end(self, outs):
        accuracy = self.test_acc.compute()
        self.log('test accuracy', accuracy)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

        lr_stepper = LrStepTracker(self.d_model, SCHEDULER_WARMUP_STEPS, 0)

        lr_scheduler = LambdaLR(opt, lr_stepper.step)

        return [opt], [lr_scheduler]
   
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif



# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory
