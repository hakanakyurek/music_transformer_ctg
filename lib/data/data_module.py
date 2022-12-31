import pytorch_lightning as pl
from .dataset import MidiDataset
from torch.utils.data import DataLoader
from lib.utilities.device import get_device 


class MidiDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, data_dir, dataset_percentage, 
                 max_seq, n_workers, random_seq=True) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.max_seq = max_seq
        self.random_seq = random_seq
        self.n_workers = n_workers
        self.data_dir = data_dir
        self.dataset_percentage = dataset_percentage

    def collate(self, batch):
        x, tgt = zip(*batch)

        return x.to(get_device()), tgt.to(get_device())

    def train_dataloader(self):
        self.train = MidiDataset(f'{self.data_dir}train/', self.max_seq, self.random_seq, self.dataset_percentage)
        return  DataLoader(self.train, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers, shuffle=True,
                           drop_last=True)

    def val_dataloader(self):
        self.val = MidiDataset(f'{self.data_dir}val/', self.max_seq, self.random_seq, self.dataset_percentage)
        return  DataLoader(self.val, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers,
                           drop_last=True)

    def test_dataloader(self):
        self.test = MidiDataset(f'{self.data_dir}test/', self.max_seq, self.random_seq, self.dataset_percentage)
        return  DataLoader(self.test, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers,
                           drop_last=True)