import os
import torch.nn as nn

import logging
from lib.utilities.logging_config import config_logging

from lib.data.data_module import MidiDataModule

from lib.model.accuracy_metric import MusicAccuracy

from lib.model.music_transformer import MusicTransformer
from lib.model.smooth_cross_entropy_loss import SmoothCrossEntropyLoss

from lib.utilities.constants import *
from lib.utilities.argument_funcs import parse_train_args, print_train_args
from lib.utilities.private_constants import wandb_key

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb


CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval accuracy"]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """
    config_logging('train')

    args = parse_train_args()
    print_train_args(args)

    os.environ['WANDB_API_KEY'] = wandb_key

    PROJECT = 'music_transformer'
    EXPERIMENT_NAME = 'test'

    SEED = 2486
    pl.seed_everything(SEED, workers=True)

    if(args.force_cpu):
        accelerator = 'cpu'
        logging.info('WARNING: Forced CPU usage, expect model to perform slower \n')
    else:
        accelerator = 'gpu'

    ##### Data Module #####
    data_module = MidiDataModule(args.batch_size, args.input_dir, 100, args.max_seq, 1)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    else:
        loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)



    ##### Model #####

    model = MusicTransformer(n_layers=args.n_layers, 
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
                                    EarlyStopping(monitor="val_loss", mode="min"),
                                    LearningRateMonitor(logging_interval='epoch')])

    trainer.fit(model=model, datamodule=data_module)
    logger.experiment.log_artifact("checkpoints/", name=f'{EXPERIMENT_NAME}_model', type='model')
    print(f'Outputted Model: {EXPERIMENT_NAME}_model')
    wandb.finish()


