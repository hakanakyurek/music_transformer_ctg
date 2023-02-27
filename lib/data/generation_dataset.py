import os
from joblib import load
import torch
from torch.utils.data import Dataset

from lib.utilities.constants import *
from lib.midi_processor.processor import decode_midi, encode_midi
from lib.data.midi_processing import process_midi, process_midi_ed
from lib.utilities.hide_prints import NoStdOut

import numpy as np

import os
import random

import pretty_midi
import music21

from tqdm import tqdm


class MidiDataset(Dataset):
    """

    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.

    """

    def __init__(self, root, arch, max_seq=2048, random_seq=False, percentage=100.0, keys=None, gedi=False):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.percentage = percentage
        self.model_arch = arch
        self.keys       = keys
        self.gedi       = gedi

        self.rng = np.random.default_rng(seed=2486)

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        # Data files hold (midi_file, encoding_file)
        self.data_files = [f for f in fs if os.path.isfile(f)] 
        if self.percentage < 100.0:
            self.data_files = self.rng.choice(self.data_files, int(self.percentage/100 * len(self.data_files)))

        self.total_data = []
        for i in range(len(self.data_files)):
            with NoStdOut():
                data_points = self.__read_encoded_midi(self.data_files[i])
                for data_point in data_points:
                    self.total_data.append(data_point)

    def __read_encoded_midi(self, data):
        # All data on cpu to allow for the Dataloader to multithread
        data = load(data)

        data_points = []

        key = data[0]
        encodings = data[1]
        f_path = data[2]

        token_key = None
        if self.keys:
            if not key in KEY_VOCAB:
                my_score: music21.stream.Score = music21.converter.parse(f_path)
                key = my_score.analyze('Krumhansl')
    
        token_key = KEY_VOCAB[key]
        # encoding --> tensor
        for enc in encodings:
            data_points.append((torch.tensor(enc, dtype=TORCH_LABEL_TYPE), token_key))
        
        return data_points

    # __len__
    def __len__(self):
        """
    
        Author: Damon Gwinn
    
        How many data files exist in the given directory
    
        """

        return len(self.total_data)

    # __getitem__
    def __getitem__(self, idx):
        """
    
        Author: Damon Gwinn
    
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
    
        """
        aug_midi, token_key = self.total_data[idx]
        if self.model_arch == 2:
            if not self.keys:
                x, tgt_input, tgt_output = process_midi_ed(aug_midi, self.max_seq, False)
            else:
                raise NotImplementedError('encoder-decoder arch isn\'t updated for key control')
            return x, tgt_input, tgt_output
        elif self.model_arch == 1:
            x, tgt = process_midi(aug_midi, self.max_seq, False, token_key)
            if self.gedi:
                return x, tgt, token_key
            else:
                return x, tgt

    def __transpose(self, midi: pretty_midi.PrettyMIDI()) -> torch.tensor:
        """
        Augments the data by shifting all of the notes to higher and/or lower pitches.
        Pitch transpositions uniformly sampled from {-3, -2, . . . , 2, 3} half-steps

        :param torch.tensor midi: preprocessed midi tensor
        :return: transposed midi tensor (if all notes could be transposed)
        """
        pitch_change = self.rng.choice([0, 5, 7]) # perfect 4th and 5th
        if pitch_change == 0:
            return midi

        for instrument in midi.instruments:
            for note in instrument.notes:
                note.pitch += pitch_change

        return midi
