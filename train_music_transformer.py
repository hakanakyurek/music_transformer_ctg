import os
import torch.nn as nn

from lib.data.generation_data_module import MidiDataModule

from lib.losses.smooth_cross_entropy_loss import SmoothCrossEntropyLoss

from lib.utilities.constants import *
from lib.utilities.argument_funcs import parse_train_args, print_train_args
from lib.utilities.private_constants import wandb_key
from lib.utilities.device import use_cuda
from lib.utilities.create_model import create_model_for_training

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import wandb

# main
def main():
    """

    Entry point. Trains a model specified by command line arguments

    """

    args = parse_train_args()
    print_train_args(args)
    vocab['size'] = VOCAB_SIZE_KEYS

    if (args.run_id and not args.checkpoint_path) or (not args.run_id and args.checkpoint_path):
        print('Run id and Checkpoint path should be given together!')
        return

    os.environ['WANDB_API_KEY'] = wandb_key

    PROJECT = 'music_transformer'
    # TODO: make it adjustable with args
    EXPERIMENT_NAME = args.experiment_name

    RUN_ID = wandb.util.generate_id() if not args.run_id else args.run_id

    SEED = 2486
    pl.seed_everything(SEED, workers=True)

    if(args.force_cpu):
        accelerator = 'cpu'
        use_cuda(False)
        print('WARNING: Forced CPU usage, expect model to perform slower \n')
    else:
        accelerator = 'gpu'

    ##### Data Module #####
    data_module = MidiDataModule(args.batch_size, args.input_dir, args.dataset_percentage, args.max_sequence, 
                                 args.n_workers, args.arch, random_seq=True, keys=args.key, cocon=args.cocon)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    else:
        loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, vocab['size'], ignore_index=TOKEN_PAD)

    ##### Model #####
    model = create_model_for_training(args, loss_func)

    ##### Checkpoint? #####
    try:
        os.makedirs(f"checkpoints/{RUN_ID}/")
    except:
        print('Checkpoint dir already exists!')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"checkpoints/{RUN_ID}/", save_top_k=1, 
                                                       save_last=True, monitor="validation loss", 
                                                       filename="best")

    ##### Init wandb #####

    wandb.init(project=PROJECT, name=EXPERIMENT_NAME, job_type="train", config=args, id=RUN_ID, resume='allow')

    ##### Logger #####

    logger = pl.loggers.WandbLogger(log_model=None)

    ##### Training #####

    trainer = pl.Trainer(accelerator=accelerator, max_epochs=args.epochs, logger=logger, 
                         callbacks=[checkpoint_callback, 
                                    EarlyStopping(monitor="validation loss", mode="min", patience=3),
                                    LearningRateMonitor(logging_interval='epoch')],
                         log_every_n_steps=10)
    if args.checkpoint_path:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model=model, datamodule=data_module)

    logger.experiment.log_artifact(f"checkpoints/{RUN_ID}/", name=f'{EXPERIMENT_NAME}_model', type='model')

    print(f'Outputted Model: {EXPERIMENT_NAME}_model')

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(e)
    finally:
        wandb.finish()
        exit(9)
 