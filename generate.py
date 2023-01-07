import torch
import os
import random

from lib.midi_processor.processor import decode_midi, encode_midi

from lib.utilities.argument_funcs import parse_generate_args, print_generate_args
from lib.data.dataset import process_midi

from lib.utilities.create_model import create_model_for_generation
from lib.utilities.constants import *
from lib.utilities.device import get_device, use_cuda

# main
def main():
    """

    Entry point. Generates music from a model specified by command line arguments

    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")

    os.makedirs(args.output_dir, exist_ok=True)

    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        print('Error: You need to set a primer!')
        return
    else:
        f = args.primer_file

    raw_mid = encode_midi(f)
    if(len(raw_mid) == 0):
        print(f"Error: No midi messages in primer file: {f}")
        return

    primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
    primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

    print(f"Using primer file: {f}")

    model = create_model_for_generation(args)

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        print('Generating...')
        rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, 
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

        f_path = os.path.join(args.output_dir, "rand.mid")
        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)




if __name__ == "__main__":
    main()
