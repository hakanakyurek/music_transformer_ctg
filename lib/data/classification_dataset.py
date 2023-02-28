import os
from joblib import load
import torch
from torch.utils.data import Dataset

from lib.utilities.constants import *
from lib.utilities.hide_prints import NoStdOut
from lib.data.midi_processing import process_midi

import numpy as np

import os
import music21



class ClassificationDataset(Dataset):
    """

    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.

    """

    def __init__(self, root, max_seq=2048, percentage=100.0, classification_task='key'):
        self.root       = root
        self.max_seq    = max_seq
        self.percentage = percentage
        self.task = classification_task

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
                if data_points is None:
                    continue
                for data_point in data_points:
                    self.total_data.append(data_point)

    def __read_encoded_midi(self, data):
        # All data on cpu to allow for the Dataloader to multithread
        data = load(data)

        data_points = []

        y = data[0]
        encodings = data[1]
        f_path = data[2]

        if self.task == 'key':
            token_key = None
            if not y in KEY_DICT:
                my_score: music21.stream.Score = music21.converter.parse(f_path)
                y = my_score.analyze('Krumhansl')
        
            token_key = KEY_DICT[y]
            # encoding --> tensor
            for enc in encodings:
                data_points.append((torch.tensor(enc, dtype=TORCH_LABEL_TYPE), 
                                   (torch.tensor([token_key], dtype=TORCH_LABEL_TYPE))))
        elif self.task == 'artist':
            y = y.split(' / ')[0]
            token_artist = ARTIST_DICT[y]
            # encoding --> tensor
            for enc in encodings:
                data_points.append((torch.tensor(enc, dtype=TORCH_LABEL_TYPE),
                                   (torch.tensor([token_artist], dtype=TORCH_LABEL_TYPE))))
        elif self.task == 'genre':
            token_genre = GENRE_DICT[y]
            if token_genre < 12:
                return None
            # encoding --> tensor
            for enc in encodings:
                data_points.append((torch.tensor(enc, dtype=TORCH_LABEL_TYPE),
                                   (torch.tensor([token_genre], dtype=TORCH_LABEL_TYPE))))

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
        x, _ = process_midi(aug_midi, self.max_seq, False,)
        return x, token_key