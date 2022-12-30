import argparse

from .constants import SEPERATOR

import logging

# parse_train_args
def parse_train_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Argparse arguments for training a model
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-input_dir", type=str, default="./dataset/e_piano", help="Folder of preprocessed and pickled midi files")
    parser.add_argument("-output_dir", type=str, default="./saved_models", help="Folder to save model weights. Saves one every epoch")
    parser.add_argument("-weight_modulus", type=int, default=1, help="How often to save epoch weights (ex: value of 10 means save every 10 epochs)")
    parser.add_argument("-print_modulus", type=int, default=1, help="How often to print train results for a batch (batch loss, learn rate, etc.)")

    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")
    parser.add_argument("--no_tensorboard", action="store_true", help="Turns off tensorboard result reporting")

    parser.add_argument("-continue_weights", type=str, default=None, help="Model weights to continue training based on")
    parser.add_argument("-continue_epoch", type=int, default=None, help="Epoch the continue_weights model was at")

    parser.add_argument("-lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("-ce_smoothing", type=float, default=None, help="Smoothing parameter for smoothed cross entropy loss (defaults to no smoothing)")
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs to use")

    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

    parser.add_argument("-dropout", type=float, default=0.1, help="Dropout rate")

    return parser.parse_args()

# print_train_args
def print_train_args(args):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Prints training arguments
    ----------
    """

    logging.info(SEPERATOR)
    logging.info(f"input_dir:{args.input_dir}")
    logging.info(f"output_dir:{args.output_dir}")
    logging.info(f"weight_modulus:{args.weight_modulus}")
    logging.info(f"print_modulus:{args.print_modulus}")
    logging.info(f"")
    logging.info(f"n_workers:{args.n_workers}")
    logging.info(f"force_cpu:{args.force_cpu}")
    logging.info(f"tensorboard:{not args.no_tensorboard}")
    logging.info(f"")
    logging.info(f"continue_weights:{args.continue_weights}")
    logging.info(f"continue_epoch:{args.continue_epoch}")
    logging.info(f"")
    logging.info(f"lr:{args.lr}")
    logging.info(f"ce_smoothing:{args.ce_smoothing}")
    logging.info(f"batch_size:{args.batch_size}")
    logging.info(f"epochs:{args.epochs}")
    logging.info(f"")
    logging.info(f"rpr:{args.rpr}")
    logging.info(f"max_sequence:{args.max_sequence}")
    logging.info(f"n_layers:{args.n_layers}")
    logging.info(f"num_heads:{args.num_heads}")
    logging.info(f"d_model:{args.d_model}")
    logging.info(f"")
    logging.info(f"dim_feedforward:{args.dim_feedforward}")
    logging.info(f"dropout:{args.dropout}")
    logging.info(SEPERATOR)
    logging.info(f"")

# parse_eval_args
def parse_eval_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Argparse arguments for evaluating a model
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_dir", type=str, default="./dataset/e_piano", help="Folder of preprocessed and pickled midi files")
    parser.add_argument("-model_weights", type=str, default="./saved_models/model.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

    parser.add_argument("-batch_size", type=int, default=2, help="Batch size to use")

    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider in the model")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

    return parser.parse_args()

# print_eval_args
def print_eval_args(args):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Prints evaluation arguments
    ----------
    """

    logging.info(SEPERATOR)
    logging.info(f"dataset_dir:{args.dataset_dir}")
    logging.info(f"model_weights:{args.model_weights}")
    logging.info(f"n_workers:{args.n_workers}")
    logging.info(f"force_cpu:{args.force_cpu}")
    logging.info(f"")
    logging.info(f"batch_size:{args.batch_size}")
    logging.info(f"")
    logging.info(f"rpr:{args.rpr}")
    logging.info(f"max_sequence:{args.max_sequence}")
    logging.info(f"n_layers:{args.n_layers}")
    logging.info(f"num_heads:{args.num_heads}")
    logging.info(f"d_model:{args.d_model}")
    logging.info(f"")
    logging.info(f"dim_feedforward:{args.dim_feedforward}")
    logging.info(SEPERATOR)
    logging.info(f"")

# parse_generate_args
def parse_generate_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Argparse arguments for generation
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, default="./dataset/e_piano/", help="Midi file to prime the generator with")
    parser.add_argument("-output_dir", type=str, default="./gen", help="Folder to write generated midi to")
    parser.add_argument("-primer_file", type=str, default=None, help="File path or integer index to the evaluation dataset. Default is to select a random index.")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

    parser.add_argument("-target_seq_length", type=int, default=1024, help="Target length you'd like the midi to be")
    parser.add_argument("-num_prime", type=int, default=256, help="Amount of messages to prime the generator with")
    parser.add_argument("-model_weights", type=str, default="./saved_models/model.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")

    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

    parser.add_argument('--temperature', type=float, default=1.0, help='Creativeness setting for the logits')
    parser.add_argument('--top_k', type=int, default=0, help='Top k for the filtering')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top p for the filtering')

    return parser.parse_args()

# print_generate_args
def print_generate_args(args):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Prints generation arguments
    ----------
    """

    logging.info(SEPERATOR)
    logging.info(f"midi_root:{args.midi_root}")
    logging.info(f"output_dir:{args.output_dir}")
    logging.info(f"primer_file:{args.primer_file}")
    logging.info(f"force_cpu:{args.force_cpu}")
    logging.info(f"")
    logging.info(f"target_seq_length:{args.target_seq_length}")
    logging.info(f"num_prime:{args.num_prime}")
    logging.info(f"model_weights:{args.model_weights}")
    logging.info(f'temperature:{args.temperature}')
    logging.info(f'top_k:{args.top_k}')
    logging.info(f'top_p:{args.top_p}')
    logging.info(f"")
    logging.info(f"rpr:{args.rpr}")
    logging.info(f"max_sequence:{args.max_sequence}")
    logging.info(f"n_layers:{args.n_layers}")
    logging.info(f"num_heads:{args.num_heads}")
    logging.info(f"d_model:{args.d_model}")
    logging.info(f"")
    logging.info(f"dim_feedforward: {args.dim_feedforward}")
    logging.info(SEPERATOR)
    logging.info(f"")

# write_model_params
def write_model_params(args, output_file):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Writes given training parameters to text file
    ----------
    """

    o_stream = open(output_file, "w")

    o_stream.write("rpr: " + str(args.rpr) + "\n")
    o_stream.write("lr: " + str(args.lr) + "\n")
    o_stream.write("ce_smoothing: " + str(args.ce_smoothing) + "\n")
    o_stream.write("batch_size: " + str(args.batch_size) + "\n")
    o_stream.write("max_sequence: " + str(args.max_sequence) + "\n")
    o_stream.write("n_layers: " + str(args.n_layers) + "\n")
    o_stream.write("num_heads: " + str(args.num_heads) + "\n")
    o_stream.write("d_model: " + str(args.d_model) + "\n")
    o_stream.write("dim_feedforward: " + str(args.dim_feedforward) + "\n")
    o_stream.write("dropout: " + str(args.dropout) + "\n")

    o_stream.close()
