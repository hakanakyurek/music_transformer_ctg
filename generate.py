import torch
import os
import random

from lib.midi_processor.processor import decode_midi, encode_midi

from lib.utilities.argument_funcs import parse_generate_args, print_generate_args

from lib.data.generation_dataset import process_midi, MidiDataset

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
    vocab['size'] = VOCAB_SIZE_KEYS if args.key else VOCAB_SIZE_NORMAL

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")

    os.makedirs(args.output_dir, exist_ok=True)

    f = args.primer_file
    
    raw_mid = encode_midi(f)
    if(len(raw_mid) == 0):
        print(f"Error: No midi messages in primer file: {f}")
        return
    raw_mid = torch.tensor(raw_mid, dtype=TORCH_LABEL_TYPE)
    primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False, token_key=args.key)
    #primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

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
