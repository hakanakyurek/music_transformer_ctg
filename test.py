import torch
import os
import music21
from tqdm import tqdm

from lib.midi_processor.processor import decode_midi, encode_midi
from lib.utilities.argument_funcs import parse_test_args, print_test_args
from lib.data.generation_dataset import process_midi
from lib.utilities.create_model import create_model_for_generation
from lib.utilities.device import get_device, use_cuda
from lib.utilities.constants import *
from lib.data.midi_processing import *

from sklearn.metrics import accuracy_score
import numpy as np

# main
def test(piece, output_dir, args):
    """

    Entry point. Generates music from a model specified by command line arguments

    """
    classes = [None, None]    

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
    if args.keys:
        my_score: music21.stream.Score = music21.converter.parse(f_path)
        key_primer = my_score.analyze('Krumhansl')
        classes[0] = key_primer

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        print('Generating...')
        rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, 
                                  temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

        f_path = os.path.join(output_dir, "rand.mid")
        rand_seq = rand_seq[0].cpu().numpy()
        decode_midi(rand_seq, file_path=f_path)
        if args.keys:
            my_score: music21.stream.Score = music21.converter.parse(f_path)
            key_rand = my_score.analyze('Krumhansl')
            classes[1] = key_rand

    return raw_mid[:len(rand_seq)], rand_seq, classes


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

    # Metrics
    overall_acc = 0
    per_piece_accuracy = []
    primer_classes = []
    output_classes = []

    # Can be None, an integer index to dataset
    if(args.primer_index is None):
        for piece in tqdm(pieces):
            output_dir = os.path.join(args.output_dir, piece)
            raw_mid, rand_seq, classes = test(args, piece, output_dir)
            p_acc = accuracy_score(raw_mid, rand_seq) 
            per_piece_accuracy.append(p_acc)
            primer_classes.append(classes[0])
            output_classes.append(classes[1])
    else:
        piece = pieces[args.primer_index]
        output_dir = os.path.join(args.output_dir, piece)
        print(f"Using primer file: {piece}")
        raw_mid, rand_seq, classes = test(args, piece, output_dir)
        p_acc = accuracy_score(raw_mid, rand_seq) 
        per_piece_accuracy.append(p_acc)
        primer_classes.append(classes[0])
        output_classes.append(classes[1])

    overall_acc = np.mean(per_piece_accuracy)

    print(per_piece_accuracy)
    print(overall_acc)