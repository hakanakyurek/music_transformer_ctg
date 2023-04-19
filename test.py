import torch
import os
import music21
from music21 import interval
from tqdm import tqdm
from joblib import load

from lib.utilities.hide_prints import NoStdOut
from lib.midi_processor.processor import decode_midi, encode_midi
from lib.utilities.argument_funcs import parse_test_args, print_test_args
from lib.data.generation_dataset import process_midi
from lib.utilities.create_model import create_model_for_generation, create_model_for_classification_test
from lib.utilities.device import get_device, use_cuda
from lib.utilities.constants import *
from lib.data.midi_processing import *

from sklearn.metrics import accuracy_score
import numpy as np
import traceback

# main
def test(piece, output_dir, args):
    """

    Entry point. Generates music from a model specified by command line arguments

    """
    classes = {'primer': None, 'algo': None, 'model': None, 'target': None}

    raw_mid = encode_midi(piece)
    if(len(raw_mid) == 0):
        print(f"Error: No midi messages in primer file: {piece}")
        return
    raw_mid = torch.tensor(raw_mid, dtype=TORCH_LABEL_TYPE)

    # Class condition to perfect 4th or perfect 5th
    if args.key:
        my_score: music21.stream.Score = music21.converter.parse(piece)
        key_primer = my_score.analyze('Krumhansl')
        classes['primer'] = KEY_DICT[str(key_primer)]
        
        intervals = [interval.Interval('P4'), interval.Interval('P5')]
        inter = np.random.choice(intervals)
        key_target = str(inter.transposePitch(key_primer.tonic))
        
        if key_target == 'D-':
            key_target = 'C#'
        if key_target == 'G#':
            key_target = 'A-'

        if key_target.isupper():
            key_target += ' major'
        else:
            key_target += ' minor'

        token_key = KEY_DICT[str(key_target)]

        classes['target'] = token_key
    else:
        token_key = None

    primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False, token_key=token_key)

    # Saving primer first
    f_path = os.path.join(output_dir, "primer.mid")
    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)
    primer = primer.to(get_device())

    # GENERATION
    generator.eval()
    with torch.set_grad_enabled(False) and NoStdOut():
        if args.cocon:
            c = torch.full((args.target_seq_length, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
            c[0] = token_key
            context = context.unsqueeze(1)
            rand_seq = generator.generate(primer[:args.num_prime], c, args.target_seq_length, 
                                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)   
        else:
            rand_seq = generator.generate(primer[:args.num_prime], args.target_seq_length, 
                                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p) 
        f_path = os.path.join(output_dir, "rand.mid")
        if args.key:
            classes['model'] = torch.argmax(classifier(rand_seq)[1]).item()

        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)        

        if args.key:
            my_score: music21.stream.Score = music21.converter.parse(f_path)
            key_rand = my_score.analyze('Krumhansl')
            classes['algo'] = KEY_DICT[str(key_rand)]
        

    return raw_mid.cpu().numpy(), rand_seq[0].cpu().numpy(), classes


if __name__ == "__main__":

    args = parse_test_args()
    print_test_args(args)

    if type(args.primer_index) is str:
        raise Exception('primer file is not accepted here, use an integer instead')

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")

    os.makedirs(args.output_dir, exist_ok=True)

    n_classes = 0
    if args.key:
        n_classes = len(KEY_DICT)

    vocab['size'] = VOCAB_SIZE_NORMAL

    classifier = create_model_for_classification_test(args, n_classes)

    vocab['size'] = VOCAB_SIZE_KEYS if args.key else VOCAB_SIZE_NORMAL

    if args.key:
        generator = create_model_for_generation(args, args.model_weights)
    else:
        generator = classifier.backbone

    pieces = []

    for subdir, dirs, files in os.walk(args.midi_root):
        for file in files:
            piece = os.path.join(subdir, file)
            piece = load(piece)[2]
            pieces.append(piece) 

    pieces.sort()

    print(f"Found {len(pieces)} pieces")

    # Metrics
    # overall_acc = 0
    # per_piece_accuracy = []
    # Key arrays
    keys_dict = {'primer': [], 'algo': [], 'model': [], 'target': []}


    # Can be None, an integer index to dataset
    if(args.primer_index is None):
        for piece in tqdm(pieces):
            try:
                output_dir = os.path.join(args.output_dir, piece.split('/')[-1])
                os.makedirs(output_dir, exist_ok=True)
                raw_mid, rand_seq, classes = test(piece, output_dir, args)
                # p_acc = accuracy_score(raw_mid, rand_seq) 
                # per_piece_accuracy.append(p_acc)
                
                keys_dict['primer'].append(classes['primer'])
                keys_dict['algo'].append(classes['algo'])
                keys_dict['model'].append(classes['model'])
                keys_dict['target'].append(classes['target'])
            except Exception as e:
                print(traceback.format_exc())
                continue
    else:
        piece = pieces[args.primer_index]
        output_dir = os.path.join(args.output_dir, piece.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)        
        print(f"Using primer file: {piece}")
        raw_mid, rand_seq, classes = test(piece, output_dir, args)
        # p_acc = accuracy_score(raw_mid, rand_seq) 
        # per_piece_accuracy.append(p_acc)
    
        keys_dict['primer'].append(classes['primer'])
        keys_dict['algo'].append(classes['algo'])
        keys_dict['model'].append(classes['model'])
        keys_dict['target'].append(classes['target'])

    # overall_acc = np.mean(per_piece_accuracy)
    # print('Sequence Accuracies')
    # print(SEPERATOR)
    # print(f'Per piece accuracies: \n {per_piece_accuracy}')
    # print(SEPERATOR)
    # print(f'Overall accuracy: {overall_acc}')
    # print(SEPERATOR)

    if args.key:
        # Check how much of the output keys are matching with the target according to the classifier
        mt_key_acc = accuracy_score(keys_dict['target'], keys_dict['model'])
        # Check how much of the output keys are matching with the target according to the algorithm
        at_key_acc = accuracy_score(keys_dict['target'], keys_dict['algo'])
        # Check how much of the output keys are matching with the target according to the algorithm
        ma_key_acc = accuracy_score(keys_dict['model'], keys_dict['algo'])
        # Check how much of the primer keys are matching with the target according to the algorithm
        # This metric is for checking whether the baseline model can continue the primer in its key
        # For cclm this doesn't makes sense as we expect it to not follow it
        pt_key_acc = accuracy_score(keys_dict['target'], keys_dict['primer'])
        print('Key Accuracies:')
        print(SEPERATOR)
        print('Classifier-Target Key Accuracy')
        print(f'Accuracy: {mt_key_acc}')
        print(SEPERATOR)
        print('Classifier-Algo Key Accuracy')
        print(f'Accuracy: {ma_key_acc}')
        print(SEPERATOR)
        print('Algorithm-Target Key Accuracy')
        print(f'Accuracy: {at_key_acc}')
        print(SEPERATOR)
        print('Primer-Target Key Accuracy')
        print(f'Accuracy: {pt_key_acc}')
