import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.data.dataset import create_datasets

from lib.model.music_transformer import MusicTransformer

from lib.utilities.constants import *
from lib.utilities.device import get_device, use_cuda
from lib.utilities.argument_funcs import parse_eval_args, print_eval_args
from lib.utilities.run_model import eval_model

import logging 
from lib.utilities.logging_config import config_logging

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Evaluates a model specified by command line arguments
    ----------
    """


    config_logging('eval')

    args = parse_eval_args()
    print_eval_args(args)

    if(args.force_cpu):
        use_cuda(False)
        logging.warning("WARNING: Forced CPU usage, expect model to perform slower")

    # Test dataset
    _, _, test_dataset = create_datasets(args.dataset_dir, args.max_sequence)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # No smoothed loss
    loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    logging.info("Evaluating:")
    model.eval()

    avg_loss, avg_acc = eval_model(model, test_loader, loss)

    logging.info(f"Avg loss: {avg_loss}")
    logging.info(f"Avg acc: {avg_acc}")
    logging.info(SEPERATOR)
    


if __name__ == "__main__":
    main()
