import argparse

from .constants import SEPERATOR

import logging

# parse_train_args
def parse_train_args():
    """

    Argparse arguments for training a model

    """

    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument("--input_dir", type=str, default="./dataset/e_piano", help="Folder of preprocessed and pickled midi files")
    parser.add_argument("--key_dir", type=str, default="./dataset/keys", help="Binary key dictionary (dumped with joblib) for each data in the dataset")
    # Environment parameters
    parser.add_argument("--n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")
    # Checkpoint parameters
    parser.add_argument('--run_id', type=str, help='The training id to resume (check wandb)')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path for training resume, resume flag should be up')
    # Training parameters
    parser.add_argument("--lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("--ce_smoothing", type=float, default=None, help="Smoothing parameter for smoothed cross entropy loss (defaults to no smoothing)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to use")
    parser.add_argument("--experiment_name", type=str, default='test', help="Name of the experiment")
    parser.add_argument('--keys', action='store_true', help='Enable training with keys')
    # Model parameters
    parser.add_argument("--arch", type=int, help='Change model architecture \n --> encoder-decoder: 1 \n --> encoder: 2')
    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("--max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    # Dataset parameters
    parser.add_argument('--dataset_percentage', type=float, default=100.0, help='Set how much of the dataset should be used')

    return parser.parse_args()

# print_train_args
def print_train_args(args):
    """

    Prints training arguments

    """

    print(SEPERATOR)
    print(f"Path parameters")
    print(f"input_dir:{args.input_dir}")
    print(f"key_dir:{args.key_dir}")
    print(f"Environment parameters")
    print(f"n_workers:{args.n_workers}")
    print(f"force_cpu:{args.force_cpu}")
    print(f"Checkpoint parameters")
    print(f"Run id:{args.run_id}")
    print(f"Checkpoint path:{args.checkpoint_path}")
    print(f"Training parameters")
    print(f"lr:{args.lr}")
    print(f"ce_smoothing:{args.ce_smoothing}")
    print(f"batch_size:{args.batch_size}")
    print(f"epochs:{args.epochs}")
    print(f"epochs:{args.experiment_name}")
    print(f'keys enabled:{args.keys}')
    print(f"Model parameters")
    print(f"rpr:{args.rpr}")
    print(f"max_sequence:{args.max_sequence}")
    print(f"n_layers:{args.n_layers}")
    print(f"num_heads:{args.num_heads}")
    print(f"d_model:{args.d_model}")
    print(f"dim_feedforward:{args.dim_feedforward}")
    print(f"dropout:{args.dropout}")
    print(f"Dataset parameters")
    print(f'dataset_percentage:{args.dataset_percentage}')
    print(SEPERATOR)
    print(f"")

# parse_generate_args
def parse_generate_args():
    """

    Argparse arguments for generation

    """

    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument("--midi_root", type=str, help="Midi file to prime the generator with")
    parser.add_argument("--output_dir", type=str, default="./output", help="Folder to write generated midi to")
    parser.add_argument("--primer_file", default=None, help="File path or integer index to the evaluation dataset. Default is to select a random index.")
    # Environment parameters
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")
    # Generation parameters
    parser.add_argument("--target_seq_length", type=int, default=1024, help="Target length you'd like the midi to be")
    parser.add_argument("--num_prime", type=int, default=256, help="Amount of messages to prime the generator with")
    parser.add_argument("--model_weights", type=str, default="./models/best.pt", help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument('--temperature', type=float, default=1.0, help='Creativeness setting for the logits')
    parser.add_argument('--top_k', type=int, default=0, help='Top k for the filtering')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top p for the filtering')
    parser.add_argument('--key', type=int, default=0, help='Enable generation with keys')
    # Model parameters
    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("--max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument("--arch", type=int, help='Change model architecture \n --> encoder-decoder: 1 \n --> encoder: 2')
    
    return parser.parse_args()

# print_generate_args
def print_generate_args(args):
    """

    Prints generation arguments

    """

    print(SEPERATOR)
    print(f'Path parameters')
    print(f"midi_root:{args.midi_root}")
    print(f"output_dir:{args.output_dir}")
    print(f"primer_file:{args.primer_file}")
    print(f'Environment parameters')
    print(f"force_cpu:{args.force_cpu}")
    print(f"Generation parameters")
    print(f"target_seq_length:{args.target_seq_length}")
    print(f"num_prime:{args.num_prime}")
    print(f"model_weights:{args.model_weights}")
    print(f'temperature:{args.temperature}')
    print(f'top_k:{args.top_k}')
    print(f'top_p:{args.top_p}')
    print(f'key:{args.key}')
    print(f"Model parameters")
    print(f"rpr:{args.rpr}")
    print(f"max_sequence:{args.max_sequence}")
    print(f"n_layers:{args.n_layers}")
    print(f"num_heads:{args.num_heads}")
    print(f"d_model:{args.d_model}")
    print(f"dim_feedforward: {args.dim_feedforward}")
    print(SEPERATOR)
    print(f"")
