from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import torch
from mel_dataset import MelDataset
from omegaconf import DictConfig


class MelDataModule(LightningDataModule):
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.train_data_dir = config.paths.train_data_script_dir
        self.val_data_dir = config.paths.val_data_script_dir
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.data_config = config.data
    
    def setup(self, stage=None):
        self.train_dataset = MelDataset(
            self.train_data_dir,
            self.data_config,            
        )
    
        self.val_dataset = MelDataset(
            self.val_data_dir,
            self.data_config
        )
    
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )