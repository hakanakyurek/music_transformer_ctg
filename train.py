import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

import logging
from lib.utilities.logging_config import config_logging

from lib.data.dataset import create_datasets

from lib.model.music_transformer import MusicTransformer
from lib.model.loss import SmoothCrossEntropyLoss

from lib.utilities.constants import *
from lib.utilities.device import get_device, use_cuda
from lib.utilities.lr_scheduling import LrStepTracker, get_lr
from lib.utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from lib.utilities.run_model import train_epoch, eval_model

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

    args = parse_train_args()
    print_train_args(args)

    config_logging()

    if(args.force_cpu):
        use_cuda(False)
        logging.info('WARNING: Forced CPU usage, expect model to perform slower \n')

    os.makedirs(args.output_dir, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter

        tensorboad_dir = os.path.join(args.output_dir, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    ##### Continuing from previous training session #####
    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            logging.error("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        logging.error("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    ##### Lr Scheduler vs static lr #####
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        train_loss_func = eval_loss_func
    else:
        train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if(epoch > BASELINE_EPOCH):
            logging.info("NEW EPOCH:", epoch+1)

            # Train
            train_epoch(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, args.print_modulus)

            logging.info("Evaluating:")
        else:
            logging.info(SEPERATOR)
            logging.info("Baseline model evaluation (Epoch 0):")

        # Eval
        train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc = eval_model(model, test_loader, eval_loss_func)

        # Learn rate
        lr = get_lr(opt)

        logging.info("Epoch:", epoch+1)
        logging.info("Avg train loss:", train_loss)
        logging.info("Avg train acc:", train_acc)
        logging.info("Avg eval loss:", eval_loss)
        logging.info("Avg eval acc:", eval_acc)
        logging.info(SEPERATOR)

        new_best = False

        if(eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch  = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if(eval_loss < best_eval_loss):
            best_eval_loss       = eval_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                logging.info("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
                logging.info("Best eval acc:", best_eval_acc, file=o_stream)
                logging.info("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
                logging.info("Best eval loss:", best_eval_loss, file=o_stream)


        if(not args.no_tensorboard):
            tensorboard_summary.add_scalar("Avg_CE_loss/train", train_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_CE_loss/eval", eval_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/train", train_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Accuracy/eval", eval_acc, global_step=epoch+1)
            tensorboard_summary.add_scalar("Learn_rate/train", lr, global_step=epoch+1)
            tensorboard_summary.flush()

        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_acc, eval_loss, eval_acc])

    # Sanity check just to make sure everything is gone
    if(not args.no_tensorboard):
        tensorboard_summary.flush()

    return


if __name__ == "__main__":
    main()
