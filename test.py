import torch
import os
import random
from tqdm import tqdm

from lib.midi_processor.processor import decode_midi, encode_midi
from lib.utilities.argument_funcs import parse_test_args, print_test_args
from lib.data.generation_dataset import process_midi
from lib.utilities.create_model import create_model_for_generation
from lib.utilities.device import get_device, use_cuda
from lib.utilities.constants import *
from lib.data.midi_processing import *


# main
def test(piece, output_dir, args):
    """

    Entry point. Generates music from a model specified by command line arguments

    """
    

    raw_mid = encode_midi(piece)
    if(len(raw_mid) == 0):
        print(f"Error: No midi messages in primer file: {piece}")
        return
    raw_mid = torch.tensor(raw_mid, dtype=TORCH_LABEL_TYPE)
    primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False, token_key=args.key)
    primer = primer.to(get_device())

    # Saving primer first
    f_path = os.path.join(output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        print('Generating...')
        rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, 
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

        f_path = os.path.join(output_dir, "rand.mid")
        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)


if __name__ == "__main__":

    args = parse_test_args()
    print_test_args(args)
    vocab['size'] = VOCAB_SIZE_KEYS if args.key else VOCAB_SIZE_NORMAL

    if type(args.primer_file) is str:
        raise Exception('primer file is not accepted here, use an integer instead')

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")

    os.makedirs(args.output_dir, exist_ok=True)

    model = create_model_for_generation(args)

    pieces = []

    for subdir, dirs, files in os.walk(args.midi_root):
        for file in files:
            piece = os.path.join(subdir, file)
            pieces.append(piece) 

    pieces = pieces.sort()

    print(f"Found {len(pieces)} pieces")

    # Can be None, an integer index to dataset
    if(args.primer_index is None):
        for piece in tqdm(pieces):
            output_dir = os.path.join(args.output_dir, piece)
            test(piece, output_dir)

    else:
        piece = pieces[args.primer_index]
        print(f"Using primer file: {piece}")
        test(args, piece)
