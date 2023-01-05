import torch
from torch.optim.lr_scheduler import LambdaLR

from lib.utilities.constants import *
from lib.utilities.device import get_device
from lib.utilities.lr_scheduling import LrStepTracker

import logging

from lib.utilities.top_p_top_k import top_k_top_p_filtering
import pytorch_lightning as pl

import torchmetrics as tm



# MusicTransformer
class MusicTransformerBase(pl.LightningModule):
    """
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    """

    def __init__(self, loss_fn, acc_metric, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False, lr=1.0):
        super(MusicTransformerBase, self).__init__()
        print(f"Creating the music transformer")

        self.train_acc = acc_metric()
        self.train_pp = tm.Perplexity(ignore_index=TOKEN_PAD)
        self.val_acc = acc_metric()
        self.val_pp = tm.Perplexity(ignore_index=TOKEN_PAD)
        self.test_acc = acc_metric()
        self.test_pp = tm.Perplexity(ignore_index=TOKEN_PAD)

    # forward
    def forward(self, x, mask=True):
        raise NotImplementedError()

    # generate
    def generate(self, primer=None, target_seq_length=1024, temperature=1.0, top_p=0.0, top_k=0):
        """
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
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
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.train_acc)

        self.log('training loss', loss)

        return loss

    def training_epoch_end(self, outs):
        accuracy = self.train_acc.compute()
        self.log('train accuracy', accuracy)
        self.log('training perplexity', self.train_pp)

    def validation_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.val_acc)
        self.log('validation loss', loss)

        return loss
    
    def validation_epoch_end(self, outs):
        accuracy = self.val_acc.compute()
        self.log('validation accuracy', accuracy)
        self.log('validation perplexity', self.val_pp)

    def test_step(self, batch, batch_idx):
        loss, y = self.step(batch, self.test_acc)
        return loss, y

    def test_epoch_end(self, outs):
        accuracy = self.test_acc.compute()
        self.log('test accuracy', accuracy)
        self.log('test perplexity', self.test_pp)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

        lr_stepper = LrStepTracker(self.d_model, SCHEDULER_WARMUP_STEPS, 0)

        lr_scheduler = LambdaLR(opt, lr_stepper.step)

        return [opt], [lr_scheduler]
