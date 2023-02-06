import music21

import argparse
import os

from tqdm import tqdm

import lib.midi_processor.processor as midi_processor

import pandas as pd
from joblib import dump


def key_custom_midi(custom_midi_root, output_dir):
    """

    Author: Corentin Nelias

    Pre-processes custom midi files that are not part of the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder. 

    """
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    

    total_count = 0
    
    pieces = []
    keys = {}

    for subdir, dirs, files in os.walk(custom_midi_root):
        for file in files:
            piece = os.path.join(subdir, file)
            pieces.append(piece)

    print(f"Found {len(pieces)} pieces")
    print("Preprocessing data...")


    for piece in tqdm(pieces):
            
        try:
            my_score: music21.stream.Score = music21.converter.parse(piece)
            key = my_score.analyze('Krumhansl')
            keys[piece] = key
        except Exception as e:
            print(e)
            continue

        total_count += 1

    dump(keys, output_dir + 'keys')
    df = pd.DataFrame(keys)
    df.to_csv(output_dir + 'keys.csv', index=[0])

    print(f"Num total: {total_count}")
    return True


# parse_args
def parse_args():
    """

    Parses arguments for preprocess_midi using argparse

    """

    parser = argparse.ArgumentParser()

    parser.add_argument("root", type=str, help="Root folder for the data.")
    parser.add_argument("--output_dir", type=str, default="./dataset/", help="CSV path for storing key info of the songs")

    return parser.parse_args()

# main
def main():
    """

    Entry point. Preprocesses maestro and saved midi to specified output folder.

    """    
    args = parse_args()
    root = args.root
    output_dir = args.output_dir

    print(f"Preprocessing midi files and saving to {output_dir}")
    key_custom_midi(root, output_dir)
    print("Done!")
    

if __name__ == "__main__":
    main()
