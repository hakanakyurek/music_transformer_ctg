import pytorch_lightning as pl
from .generation_dataset import MidiDataset
from torch.utils.data import DataLoader
import torch

class MidiDataModule(pl.LightningDataModule):
    
    def __init__(self, batch_size, data_dir, dataset_percentage, 
                 max_seq, n_workers, arch, random_seq=True, keys=None, gedi=None, cocon=None) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.max_seq = max_seq
        self.random_seq = random_seq
        self.n_workers = n_workers
        self.data_dir = data_dir
        self.dataset_percentage = dataset_percentage
        self.model_arch = arch
        self.keys = keys
        self.gedi = gedi
        self.cocon = cocon

    def collate(self, batch):
        if self.model_arch == 2:
            x, tgt_input, tgt_output = zip(*batch)

            x = torch.stack(x)
            tgt_input = torch.stack(tgt_input)
            tgt_output = torch.stack(tgt_output)
            
            return x, tgt_input, tgt_output
        elif self.model_arch == 1:
            model_input = list(zip(*batch))
            for i in range(len(model_input)):
                model_input[i] = torch.stack(model_input[i])
            
            return model_input

    def train_dataloader(self):
        self.train = MidiDataset(f'{self.data_dir}train/', self.model_arch, self.max_seq, self.random_seq, 
                                 self.dataset_percentage, self.keys, cocon=self.cocon)
        print('Train dataset size:', len(self.train))
        return  DataLoader(self.train, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers, shuffle=True,
                           drop_last=True)

    def val_dataloader(self):
        self.val = MidiDataset(f'{self.data_dir}val/', self.model_arch, self.max_seq, self.random_seq, 
                               self.dataset_percentage, self.keys, cocon=self.cocon)
        print('Val dataset size:', len(self.val))
        return  DataLoader(self.val, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers,
                           drop_last=True)

    def test_dataloader(self):
        self.test = MidiDataset(f'{self.data_dir}test/', self.model_arch, self.max_seq, self.random_seq, 
                                self.dataset_percentage, self.keys, cocon=self.cocon)
        print('Test dataset size:', len(self.test))
        return  DataLoader(self.test, batch_size=self.batch_size, 
                           collate_fn=self.collate,
                           num_workers=self.n_workers,
                           drop_last=True)