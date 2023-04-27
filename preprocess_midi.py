import argparse
import os
from joblib import dump, load
import json
import random

from tqdm import tqdm

from lib.midi_processor.processor import encode_midi, decode_midi
import music21
import pretty_midi

import pandas as pd
import numpy as np


def key_custom_midi(pieces, output_dir):
    """

    Author: Corentin Nelias

    Pre-processes custom midi files that are not part of the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder. 

    """
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    

    total_count = 0
    
    keys = {}

    for piece in tqdm(pieces):
            
        try:
            my_score: music21.stream.Score = music21.converter.parse(piece)
            key = my_score.analyze('Krumhansl')
            keys[piece] = str(key)
        except Exception as e:
            print(e)
            continue

        total_count += 1

    dump(keys, output_dir + 'keys')

    print(f"Num total: {total_count}")
    return True


def time_split(mid, enc, max_seq, task='keys_ctrl', full_seq=False):
    encodings = []
    # Get the end time of the whole midi
    max_end_time = mid.get_end_time()
    # Decode back the encoding
    # Read time for max_seq - 1 tokens if we are using keys
    max_seq = max_seq if not task == 'keys_ctrl' else max_seq - 1

    time_check_midi = decode_midi(enc[0:max_seq])

    # Get the duration for clip
    duration = time_check_midi.get_end_time()
    if duration <= 1:
        return None

    if full_seq:

        start_time = 0
        while start_time < max_end_time:
            # Get the end time
            end_time = start_time + duration
            
            if duration != max_end_time:
                # Encode the clipped part
                enc = encode_midi(mid, start_time, end_time)
                start_time += (duration - duration / 10)

            else:
                # Midi is represented with less then max_seq tokens
                start_time = end_time + 1

            encodings.append(enc)

    else:
        if duration != max_end_time:
            # Get a start time, max_end_time should be equal to duration in worst case
            start_time = random.uniform(0, max_end_time - duration)
            # Ensure the start time is at least 0
            start_time = 0 if start_time < 0 else start_time
            # Get the end time
            end_time = start_time + duration
            # Encode the clipped part
            enc = encode_midi(mid, start_time, end_time)

        encodings.append(enc)

    return encodings


def prep_custom_midi(custom_midi_root, output_dir, valid_p = 0.1, test_p = 0.2, max_seq=1024, 
                     task='keys_ctrl', full_seq=False, key_distro=None):
    """

    Author: Corentin Nelias

    Pre-processes custom midi files that are not part of the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder. 

    """

    if full_seq:
        output_dir = os.path.join(output_dir, "full_seq", task)
    else:
        if key_distro:
            output_dir = os.path.join(output_dir, "random_seq_with_equal_keys", task) 
            key_distro = load(key_distro)
        else:
            output_dir = os.path.join(output_dir, "random_seq", task) 

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0
    
    key_distro_max = max(key_distro.values())

    pieces = []

    for subdir, dirs, files in os.walk(custom_midi_root):
        for file in files:
            piece = os.path.join(subdir, file)
            pieces.append(piece)

    print(f"Found {len(pieces)} pieces")
    print("Preprocessing data...")

    artist_df = pd.read_csv(os.path.join(custom_midi_root, 'maestro-v3.0.0/maestro-v3.0.0.csv'))

    for piece in tqdm(pieces):
        # print(piece)
        #deciding whether the data should be part of train, valid or test dataset
        is_train = True if random.random() > valid_p else False
        if not is_train:
            is_valid = True if random.random() > test_p else False
        if is_train:
            split_type  = "train"
        elif is_valid:
            split_type = "validation"
        else:
            split_type = "test"
            
        f_name = piece.split('/')[-1].split('.')[0] + ".pickle"
        try:
            # Key operations
            
            if task == 'artist':
                # Only for maestro
                artist = artist_df[artist_df['midi_filename'].str.contains(piece.split('/')[-1])]['canonical_composer']
                if len(artist) != 0:
                    y = artist.iloc[0]
                else:
                    # Data is not in maestro
                    continue
            elif task == 'genre':
                dataset = piece.split('/')[2]
                if dataset == 'maestro-v3.0.0':
                    y = 'classical'
                elif dataset == 'vgmidi':
                    y = 'soundtracks'
                elif dataset == 'piano-midi':
                    y = 'classical'
                elif dataset == 'adl-piano-midi':
                    y = piece.split('/')[3].lower()
                else:
                    continue
            else:
                y = None

            my_score: music21.stream.Score = music21.converter.parse(piece)
            midi_key = my_score.analyze('Krumhansl')
            if task == 'key' or task == 'keys_ctrl':
                y = midi_key

            mid = pretty_midi.PrettyMIDI(midi_file=piece)
            earliest_note_start = min(note.start for inst in mid.instruments for note in inst.notes)
            # Subtract the earliest note start time from the start time of each note
            for inst in mid.instruments:
                [setattr(note, 'start', note.start - earliest_note_start) for note in inst.notes]
                [setattr(note, 'end', note.end - earliest_note_start) for note in inst.notes]
            enc = encode_midi(mid)
            if not full_seq:
                n_encodings = round(key_distro_max / key_distro[str(midi_key)])
                random_max_seq = lambda: np.random.randint(max_seq)
            else:
                n_encodings = 1
                random_max_seq = lambda: max_seq

            for i in range(n_encodings):
                encodings = time_split(mid, enc, random_max_seq(), task, full_seq)

            if encodings is None:
                continue     

        except Exception as e:
            print(e)
            continue
        if(split_type == "train"):
            o_file = os.path.join(train_dir, f_name)
            train_count += 1
        elif(split_type == "validation"):
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif(split_type == "test"):
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        
        dump((str(y), encodings, piece), o_file)

        total_count += 1

    print(f"Num Train: {train_count}")
    print(f"Num Val: {val_count}")
    print(f"Num Test: {test_count}")
    return True


# parse_args
def parse_args():
    """

    Parses arguments for preprocess_midi using argparse

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("root", type=str, help="Root folder for the data.")
    parser.add_argument("--output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into.")
    parser.add_argument('--max_seq', type=int, default=1024, help='Max sequence length for each data')
    parser.add_argument("--task", type=str, default='normal_lm', help="What should be the class data: key, artist(for maestro), genre")
    parser.add_argument("--full_seq", action='store_true', help="Encode full sequence data")
    parser.add_argument("--key_distro", type=str, help="key distribution for random splitting")
    

    return parser.parse_args()

# main
def main():
    """

    Entry point. Preprocesses maestro and saved midi to specified output folder.

    """    
    args = parse_args()
    root = args.root
    output_dir = args.output_dir

    random.seed(10)

    print(f"Preprocessing midi files and saving to {output_dir}")
    prep_custom_midi(root, output_dir, max_seq=args.max_seq, task=args.task, full_seq=args.full_seq, key_distro=args.key_distro)
    print("Done!")
    

if __name__ == "__main__":
    main()
