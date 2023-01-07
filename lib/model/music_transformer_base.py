import torch
from torch.optim.lr_scheduler import LambdaLR

from lib.utilities.constants import *
from lib.utilities.lr_scheduling import LrStepTracker

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

    def __init__(self, acc_metric):
        super(MusicTransformerBase, self).__init__()
        print(f"Creating the music transformer")

        if acc_metric is not None:
            self.train_acc = acc_metric()
            self.train_pp = tm.Perplexity(ignore_index=TOKEN_PAD)
            self.val_acc = acc_metric()
            self.val_pp = tm.Perplexity(ignore_index=TOKEN_PAD)
            self.test_acc = acc_metric()
            self.test_pp = tm.Perplexity(ignore_index=TOKEN_PAD)

    # forward
    def forward(self, x, mask=True):
        raise NotImplementedError()

    def generate(self, x):
        raise NotImplementedError()

    def step(self, batch, acc_metric, pp_metric):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.train_acc, self.train_pp)

        self.log('training loss', loss)

        return loss

    def training_epoch_end(self, outs):
        accuracy = self.train_acc.compute()
        self.log('train accuracy', accuracy)
        self.log('training perplexity', self.train_pp)

    def validation_step(self, batch, batch_idx):
        loss, _ = self.step(batch, self.val_acc, self.val_pp)
        self.log('validation loss', loss)

        return loss
    
    def validation_epoch_end(self, outs):
        accuracy = self.val_acc.compute()
        self.log('validation accuracy', accuracy)
        self.log('validation perplexity', self.val_pp)

    def test_step(self, batch, batch_idx):
        loss, y = self.step(batch, self.test_acc, self.test_pp)
        return loss, y

    def test_epoch_end(self, outs):
        accuracy = self.test_acc.compute()
        self.log('test accuracy', accuracy)
        self.log('test perplexity', self.test_pp)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

        lr_stepper = LrStepTracker(self.d_model, SCHEDULER_WARMUP_STEPS, 1)

        lr_scheduler = LambdaLR(opt, lr_stepper.step)

        return [opt], [lr_scheduler]

    def metric_update(self, acc_metric, pp_metric, y, y_star):
        acc_metric.update(y, y_star)
        pp_metric.update(y_star, y)
