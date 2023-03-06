import pytorch_lightning as pl
from .classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader
import torch
from lib.utilities.constants import TORCH_FLOAT

class ClassificationDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, data_dir, dataset_percentage, max_seq, n_workers, task, n_classes) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.max_seq = max_seq
        self.n_workers = n_workers
        self.data_dir = data_dir
        self.dataset_percentage = dataset_percentage
        self.task = task
        self.n_classes = n_classes

    def collate(self, batch):

        x, y = zip(*batch)

        x = torch.stack(x)
        y = torch.stack(y)
        
        return x, y[:, 0]

    def train_dataloader(self):
        self.train = ClassificationDataset(f'{self.data_dir}train/', self.max_seq, self.dataset_percentage, self.task)
        print('Train dataset size:', len(self.train))
        return  DataLoader(self.train, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers, shuffle=True,
                           drop_last=True)

    def val_dataloader(self):
        self.val = ClassificationDataset(f'{self.data_dir}val/', self.max_seq, self.dataset_percentage, self.task)
        print('Val dataset size:', len(self.val))
        return  DataLoader(self.val, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers,
                           drop_last=True)

    def test_dataloader(self):
        self.test = ClassificationDataset(f'{self.data_dir}test/', self.max_seq, self.dataset_percentage, self.task)
        print('Test dataset size:', len(self.test))
        return  DataLoader(self.test, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers)