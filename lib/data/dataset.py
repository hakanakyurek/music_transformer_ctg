import os
from joblib import load
import torch
from torch.utils.data import Dataset

from lib.utilities.constants import *
from lib.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, decode_midi, encode_midi
from lib.data.midi_processing import process_midi, process_midi_ed

import numpy as np

import os, sys
import random

import pretty_midi

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class MidiDataset(Dataset):
    """

    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.

    """

    def __init__(self, root, arch, max_seq=2048, random_seq=False, percentage=100.0):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.percentage = percentage
        self.model_arch = arch

        self.rng = np.random.default_rng(seed=2486)

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        # Data files hold (midi_file, encoding_file)
        self.data_files = [f for f in fs if os.path.isfile(f)] 
        if self.percentage < 100.0:
            self.data_files = self.rng.choice(self.data_files, int(self.percentage/100 * len(self.data_files)))

        # self.encoded_midis = [self.read_encoded_midi(idx) for idx in range(len(self.data_files))]

    def __read_encoded_midi(self, idx):
        # All data on cpu to allow for the Dataloader to multithread
        mid_enc = self.data_files[idx]
        i_stream = load(mid_enc)
        # Load midi
        mid_path = i_stream[0]
        mid = pretty_midi.PrettyMIDI(midi_file=mid_path)
        # Get encoding
        enc = i_stream[1]
        # Get the end time of the whole midi
        max_end_time = mid.get_end_time()
        # encoding --> tensor
        encoded_mid = torch.tensor(enc, dtype=TORCH_LABEL_TYPE)
        # Decode back the encoding
        decoded_mid = decode_midi(encoded_mid[0:self.max_seq])
        # Get the duration for clip
        duration = decoded_mid.get_end_time()
        if duration != max_end_time:
            # Get a start time, max_end_time should be equal to duration in worst case
            start_time = random.uniform(0, max_end_time - duration)
            # Ensure the start time is at least 0
            start_time = 0 if start_time < 0 else start_time
            # Get the end time
            end_time = start_time + duration
            # Augmentation
            # decoded_mid = self.__transpose(decoded_mid)
            # Encode the clipped part
            encoded_mid = encode_midi(mid, start_time, end_time)
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
        with HiddenPrints():
            aug_midi = self.__read_encoded_midi(idx)
        if self.model_arch == 2:
            x, tgt_input, tgt_output = process_midi_ed(aug_midi, self.max_seq, self.random_seq)
            return x, tgt_input, tgt_output
        elif self.model_arch == 1:
            x, tgt = process_midi(aug_midi, self.max_seq, self.random_seq)
            return x, tgt


    def __transpose(self, midi: torch.tensor) -> torch.tensor:
        """
        Augments the data by shifting all of the notes to higher and/or lower pitches.
        Pitch transpositions uniformly sampled from {-3, -2, . . . , 2, 3} half-steps

        :param torch.tensor midi: preprocessed midi tensor
        :return: transposed midi tensor (if all notes could be transposed)
        """
        pitch_change = self.rng.choice([-3, -2, -1, 0, 1, 2, 3])

        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)

        aug_midi = torch.clone(midi)

        for i in range(len(aug_midi)):
            final_pitch = aug_midi[i] + pitch_change
            if final_pitch in range_note_on and aug_midi[i] in range_note_on:
                aug_midi[i] += pitch_change
            elif final_pitch in range_note_off and aug_midi[i] in range_note_off:
                aug_midi[i] += pitch_change
            else:
                # Either we transpose every note or none
                del aug_midi
                return midi

        del midi
        return aug_midi
