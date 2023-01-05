import os
import torch.nn as nn

import logging
from lib.utilities.logging_config import config_logging

from lib.data.data_module import MidiDataModule

from lib.metrics.accuracy_metric import MusicAccuracy

from lib.model.music_transformer_ed import MusicTransformerEncoderDecoder
from lib.losses.smooth_cross_entropy_loss import SmoothCrossEntropyLoss

from lib.utilities.constants import *
from lib.utilities.argument_funcs import parse_train_args, print_train_args
from lib.utilities.private_constants import wandb_key
from lib.utilities.device import use_cuda

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb


# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_args()

    torch.autograd.set_detect_anomaly(True)
    os.environ['WANDB_API_KEY'] = wandb_key

    PROJECT = 'music_transformer'
    EXPERIMENT_NAME = 'test'

    SEED = 2486
    pl.seed_everything(SEED, workers=True)

    if(args.force_cpu):
        accelerator = 'cpu'
        use_cuda(False)
        print('WARNING: Forced CPU usage, expect model to perform slower \n')
    else:
        accelerator = 'gpu'

    ##### Data Module #####
    data_module = MidiDataModule(args.batch_size, args.input_dir, args.dataset_percentage, args.max_sequence, args.n_workers)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    else:
        loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    ##### Model #####

    model = MusicTransformerEncoderDecoder(n_layers=args.n_layers, 
                             num_heads=args.num_heads,
                             d_model=args.d_model, 
                             dim_feedforward=args.dim_feedforward, 
                             dropout=args.dropout,
                             max_sequence=args.max_sequence, 
                             rpr=args.rpr, 
                             acc_metric=MusicAccuracy, 
                             loss_fn=loss_func,
                             lr=LR_DEFAULT_START)

    ##### Checkpoint? #####

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoints", save_top_k=1, 
                                                       save_last=True, monitor="validation loss", 
                                                       filename="best")

    ##### Init wandb #####

    wandb.init(project=PROJECT, name=EXPERIMENT_NAME, job_type="train")

    ##### Logger #####

    logger = pl.loggers.WandbLogger(log_model=None)

    ##### Training #####

    trainer = pl.Trainer(accelerator=accelerator, max_epochs=args.epochs, logger=logger, 
                         callbacks=[checkpoint_callback, 
                                    EarlyStopping(monitor="validation loss", mode="min", patience=3),
                                    LearningRateMonitor(logging_interval='epoch')],
                         log_every_n_steps=10,
                         check_val_every_n_epoch=3)

    trainer.fit(model=model, datamodule=data_module)
    logger.experiment.log_artifact("checkpoints/", name=f'{EXPERIMENT_NAME}_model', type='model')
    print(f'Outputted Model: {EXPERIMENT_NAME}_model')
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(e)

