import torch
import time

from .constants import *
from .device import get_device
from .lr_scheduling import get_lr

from lib.data.dataset import compute_accuracy

import logging


# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())

        y = model(x)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        out = loss.forward(y, tgt)

        out.backward()
        out.detach()
        opt.step()

        if(lr_scheduler is not None):
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if((batch_num+1) % print_modulus == 0):
            logging.info(SEPERATOR)
            logging.info(f"Epoch {cur_epoch} Batch {batch_num+1}/{len(dataloader)}")
            logging.info(f"LR: {get_lr(opt)}")
            logging.info(f"Train loss: {float(out)}")
            
            logging.info(f"Time (s): {time_took}")
            logging.info(SEPERATOR)
            

    return

# eval_model
def eval_model(model, dataloader, loss):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        for batch in dataloader:
            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y = model(x)

            sum_acc += float(compute_accuracy(y, tgt))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc
