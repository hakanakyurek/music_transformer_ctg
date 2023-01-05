import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import logging

from lib.utilities.constants import *
from lib.utilities.device import cpu_device

from lib.midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF

import numpy as np

SEQUENCE_START = 0

class MidiDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True, percentage=100.0):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.percentage = percentage

        self.rng = np.random.default_rng(seed=2486)

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

        self.data_files = self.rng.choice(self.data_files, int(self.percentage/100 * len(self.data_files)))

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE)
        i_stream.close()
        aug_midi = self.__transpose(raw_mid)
        x, tgt_input, tgt_output = process_midi_ed(aug_midi, self.max_seq, self.random_seq)

        return x, tgt_input, tgt_output

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
                return midi
        
        return aug_midi

# process_midi
def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    # Shift to the right by one
    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # logging.info("x:",x)
    # logging.info("tgt:",tgt)

    return x, tgt

# process_midi for ed arch
def process_midi_ed(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt_input = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)
    tgt_output = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE)

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq for ed arch

    if(raw_len == 0):
        return x, tgt_input, tgt_output

    # Shift to the right by one
    if(raw_len < full_seq):
        x[:raw_len] = raw_mid
        
        tgt_input[0] = TOKEN_START
        tgt_input[1:raw_len] = raw_mid[1:raw_len]

        tgt_output[:raw_len - 1] = raw_mid[:raw_len - 1]
        tgt_output[-1] = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        
        tgt_input[0] = TOKEN_START
        tgt_input[1:full_seq - 1] = data[1:full_seq - 1]

        tgt_output[:full_seq - 2] = data[:full_seq - 2]
        tgt_output[-1] = TOKEN_END


    # logging.info("x:",x)
    # logging.info("tgt:",tgt)

    return x, tgt_input, tgt_output

# create_epiano_datasets
def create_datasets(dataset_root, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = MidiDataset(train_root, max_seq, random_seq)
    val_dataset = MidiDataset(val_root, max_seq, random_seq)
    test_dataset = MidiDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset

# compute_epiano_accuracy
def compute_accuracy(out, tgt):
    """
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    """

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc

