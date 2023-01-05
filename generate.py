import torch
import os
import random

from lib.midi_processor.processor import decode_midi, encode_midi

from lib.utilities.argument_funcs import parse_generate_args, print_generate_args
from lib.model.music_transformer import MusicTransformer
from lib.data.dataset import process_midi, create_datasets

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

    # Grabbing dataset if needed
    _, _, dataset = create_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print(f"Using primer index: {idx} ( {dataset.data_files[idx]} )")

    else:
        raw_mid = encode_midi(f)
        if(len(raw_mid) == 0):
            print(f"Error: No midi messages in primer file: {f}")
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print(f"Using primer file: {f}")

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights, map_location=get_device()))

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        print("RAND DIST")
        rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, 
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

        f_path = os.path.join(args.output_dir, "rand.mid")
        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)




if __name__ == "__main__":
    main()
