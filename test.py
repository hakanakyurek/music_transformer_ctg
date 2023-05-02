import torch
import os
import music21
from music21 import interval, converter, scale, note, chord
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
from torchtext.data.metrics import bleu_score
from torchmetrics.text.rouge import  ROUGEScore


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
    my_score: music21.stream.Score = music21.converter.parse(piece)
    key_primer = my_score.analyze('Krumhansl')
    classes['primer'] = KEY_DICT[str(key_primer)]
    # Class condition to perfect 4th or perfect 5th
    if False:
        intervals = [interval.Interval('P4'), interval.Interval('P5'), interval.Interval('P1')]
        relative_intervals = [interval.Interval('M6'), interval.Interval('P1')]
        inter = np.random.choice(intervals)
        relative_inter = np.random.choice(relative_intervals)
        key_target = str(inter.transposePitch(key_primer.tonic))
        key_target = str(relative_inter.transposePitch(key_primer.tonic))

        if key_target.isupper():
            if relative_intervals == interval.Interval('P1'):
                key_target += ' major'
            else:
                key_target = key_target.lower()
                key_target += ' minor'
        else:
            if relative_intervals == interval.Interval('P1'):
                key_target += ' minor'
            else:
                key_target = key_target.upper()
                key_target += ' major'

        tonic_name = key_target.split(' ')[0]
        if tonic_name == 'D-':
            key_target = 'C# major'
        if tonic_name == 'G#':
            key_target = 'A- major'
        if tonic_name == 'a-':
            key_target = 'g# minor'
        if tonic_name == 'a#':
            key_target = 'b- minor'
        if tonic_name == 'd#':
            key_target = 'e- minor'
        token_key = KEY_DICT[str(key_target)]

    else:
        key_target = str(key_primer)
        token_key = KEY_DICT[str(key_primer)]

    classes['target'] = token_key

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
            c = c.unsqueeze(0)

            rand_seq = generator.generate(primer[:args.num_prime], c, args.target_seq_length, 
                                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)   
        else:
            rand_seq = generator.generate(primer[:args.num_prime], args.target_seq_length, 
                                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p) 
        f_path = os.path.join(output_dir, "rand.mid")
        classes['model'] = torch.argmax(classifier(rand_seq)[1]).item()

        decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)        

        my_score: music21.stream.Score = music21.converter.parse(f_path)
        key_rand = my_score.analyze('Krumhansl')
        classes['algo'] = KEY_DICT[str(key_rand)]
        
    print(classes)
    return raw_mid.cpu().numpy(), rand_seq[0].cpu().numpy(), classes, key_target


if __name__ == "__main__":

    args = parse_test_args()
    print_test_args(args)

    if type(args.primer_index) is str:
        raise Exception('primer file is not accepted here, use an integer instead')

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")

    os.makedirs(args.output_dir, exist_ok=True)

    n_classes = len(KEY_DICT)

    vocab['size'] = VOCAB_SIZE_KEYS

    classifier = create_model_for_classification_test(args, n_classes)

    generator = create_model_for_generation(args, args.model_weights)

    pieces = []

    for subdir, dirs, files in os.walk(args.midi_root):
        for file in files:
            piece = os.path.join(subdir, file)
            piece = load(piece)[2]
            pieces.append(piece) 

    pieces.sort()

    print(f"Found {len(pieces)} pieces")

    # Key arrays
    keys_dict = {'primer': [], 'algo': [], 'model': [], 'target': []}
    bleu_scores = []
    note_persistency_scores = []
    rouge_scores = []

    rouge_scorer = ROUGEScore()

    # Can be None, an integer index to dataset
    if(args.primer_index is None):
        for piece in tqdm(pieces):
            try:
                output_dir = os.path.join(args.output_dir, piece.split('/')[-1])
                os.makedirs(output_dir, exist_ok=True)
                raw_mid, rand_seq, classes, key_target = test(piece, output_dir, args)
                # p_acc = accuracy_score(raw_mid, rand_seq) 
                # per_piece_accuracy.append(p_acc)
                
                keys_dict['primer'].append(classes['primer'])
                keys_dict['algo'].append(classes['algo'])
                keys_dict['model'].append(classes['model'])
                keys_dict['target'].append(classes['target'])

                # Get the notes of the target key

                output_file = os.path.join(output_dir, "rand.mid")
                midi_file = converter.parse(output_file)
                
                target_scale = scale.MajorScale(key_target.split(' ')[0])
                notes_in_scale = [str(p)[:-1][:2] for p in target_scale.getPitches()][:-1]
                # print(key_target, notes_in_scale)
                total_notes_count = 0
                scale_notes_count = 0

                for ele in midi_file.flat.getElementsByClass(['Note', 'Chord']):
                    if isinstance(ele, note.Note):
                        if str(ele.pitch.name) in notes_in_scale:
                            scale_notes_count += 1
                        total_notes_count += 1
                    elif isinstance(ele, chord.Chord):
                        for pitch in ele.pitches:
                            if str(pitch.name) in notes_in_scale:
                                scale_notes_count += 1
                            total_notes_count += 1
                #print(scale_notes_count / total_notes_count)
                note_persistency_scores.append(scale_notes_count / total_notes_count)

                # add bleu score, generation vs original
                bleu = bleu_score(np.expand_dims(rand_seq.astype(str), axis=0).tolist(), 
                                  np.expand_dims(np.expand_dims(raw_mid[:len(rand_seq)].astype(str), axis=0), axis=0).tolist())
                #print(bleu)
                bleu_scores.append(bleu)
                rouge = rouge_scorer(' '.join(rand_seq.astype(str).tolist()), 
                                   ' '.join(raw_mid[:len(rand_seq)].astype(str).tolist()))
                
                rouge_scores.append(rouge)

            except Exception as e:
                print(traceback.format_exc())
                continue
    else:
        piece = pieces[args.primer_index]
        output_dir = os.path.join(args.output_dir, piece.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)        
        print(f"Using primer file: {piece}")
        raw_mid, rand_seq, classes, key_target = test(piece, output_dir, args)
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
    with open(output_dir + 'out.txt', 'w') as f:
        print('Key Accuracies:', file=f)
        print(SEPERATOR, file=f)
        print('Classifier-Target Key Accuracy', file=f)
        print(f'Accuracy: {mt_key_acc}', file=f)
        print(SEPERATOR, file=f)
        print('Classifier-Algo Key Accuracy', file=f)
        print(f'Accuracy: {ma_key_acc}', file=f)
        print(SEPERATOR, file=f)
        print('Algorithm-Target Key Accuracy', file=f)
        print(f'Accuracy: {at_key_acc}', file=f)
        print(SEPERATOR, file=f)
        print('Primer-Target Key Accuracy', file=f)
        print(f'Accuracy: {pt_key_acc}', file=f)
        print(SEPERATOR, file=f)
        print('Mean Bleu Score', file=f)
        print(f'Score: {np.mean(bleu_scores)}', file=f)
        print(SEPERATOR, file=f)
        print('Mean ROUGE Score', file=f)
        print(f'Score: {np.mean(rouge_scores)}', file=f)
        print(SEPERATOR, file=f)
        print('Mean Note Persistency', file=f)
        print(f'Score: {np.mean(note_persistency_scores)}', file=f)
