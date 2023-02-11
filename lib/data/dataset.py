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


class MidiDataset(Dataset):
    """

    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.

    """

    def __init__(self, root, arch, max_seq=2048, random_seq=False, percentage=100.0, keys=None):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.percentage = percentage
        self.model_arch = arch
        self.keys       = load(keys) if keys else keys

        self.rng = np.random.default_rng(seed=2486)

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        # Data files hold (midi_file, encoding_file)
        self.data_files = [f for f in fs if os.path.isfile(f)] 
        if self.percentage < 100.0:
            self.data_files = self.rng.choice(self.data_files, int(self.percentage/100 * len(self.data_files)))

        # Sanity check for keys
        if keys:
            temp = []
            for data in self.data_files:
                mid_path = load(data)[0]
                if mid_path in self.keys.keys():
                    temp.append(data)
            self.data_files = temp

        # self.encoded_midis = [self.read_encoded_midi(idx) for idx in range(len(self.data_files))]

    def __read_encoded_midi(self, idx):
        # All data on cpu to allow for the Dataloader to multithread
        mid_enc = self.data_files[idx]
        i_stream = load(mid_enc)
        # Load midi
        mid_path = i_stream[0]
        mid = pretty_midi.PrettyMIDI(midi_file=mid_path)
        # Key operations
        if self.keys:
            key = self.keys[mid_path]
            if not key in KEY_DICT:
                my_score: music21.stream.Score = music21.converter.parse(mid_path)
                key = my_score.analyze('Krumhansl')

            token_key = KEY_DICT[key]        
        # Get encoding
        enc = i_stream[1]
        # Get the end time of the whole midi
        max_end_time = mid.get_end_time()
        # Decode back the encoding
        # Read time for max_seq - 1 tokens if we are using keys
        max_seq = self.max_seq if not self.keys else self.max_seq - 1
        decoded_mid = decode_midi(enc[0:max_seq])
        # Get the duration for clip
        duration = decoded_mid.get_end_time()
        # If the midi is shorter than max sequence
        if duration != max_end_time:
            if self.random_seq:
                # Get a start time, max_end_time should be equal to duration in worst case
                start_time = random.uniform(0, max_end_time - duration)
            else:
                start_time = 0
            # Ensure the start time is at least 0
            start_time = 0 if start_time < 0 else start_time
            # Get the end time
            end_time = start_time + duration
            # Augmentation
            # decoded_mid = self.__transpose(decoded_mid)
            # Encode the clipped part
            enc = encode_midi(mid, start_time, end_time)
        # encoding --> tensor
        encoded_mid = torch.tensor(enc, dtype=TORCH_LABEL_TYPE)
        if self.keys:
            token_key = torch.tensor([token_key], dtype=TORCH_LABEL_TYPE)
            encoded_mid = torch.cat((token_key, encoded_mid), dim=0)
        return encoded_mid

    # __len__
    def __len__(self):
        """
    
        Author: Damon Gwinn
    
        How many data files exist in the given directory
    
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
    
        Author: Damon Gwinn
    
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
    
        """
        with NoStdOut():
            aug_midi = self.__read_encoded_midi(idx)
        if self.model_arch == 2:
            if not self.keys:
                x, tgt_input, tgt_output = process_midi_ed(aug_midi, self.max_seq, False)
            else:
                raise NotImplementedError('encoder-decoder arch isn\'t updated for key control')
            return x, tgt_input, tgt_output
        elif self.model_arch == 1:
            x, tgt = process_midi(aug_midi, self.max_seq, False)
            return x, tgt


    def __transpose(self, midi: pretty_midi.PrettyMIDI()) -> torch.tensor:
        """
        Augments the data by shifting all of the notes to higher and/or lower pitches.
        Pitch transpositions uniformly sampled from {-3, -2, . . . , 2, 3} half-steps

        :param torch.tensor midi: preprocessed midi tensor
        :return: transposed midi tensor (if all notes could be transposed)
        """
        pitch_change = self.rng.choice([-3, -2, -1, 0, 1, 2, 3])

        midi.transpose(pitch_change)

        return midi
