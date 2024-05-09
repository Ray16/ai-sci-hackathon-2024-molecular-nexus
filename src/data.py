import torch
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

# pytorch geometric graph dataset
class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

# pytorch lightning data module
class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int = 32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset,[0.8,0.1,0.1], generator = torch.Generator().manual_seed(42))

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        torch.save(train_loader, './dataloader/train_dataloader.pth')
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        torch.save(val_loader, './dataloader/val_dataloader.pth')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        torch.save(test_loader, './dataloader/test_dataloader.pth')
        return test_loader
    
    def collate_fn(self, batch):
        batch = InMemoryDataset.collate(batch)
        return batch
    
# pytorch lightning data module
class PredGraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size: int = 32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
    '''
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        torch.save(train_loader, './dataloader/train_dataloader.pth')
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        torch.save(val_loader, './dataloader/val_dataloader.pth')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
    '''
    def predict_dataloader(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True,collate_fn=self.collate_fn)
        return dataloader
                 
    def collate_fn(self, batch):
        batch = InMemoryDataset.collate(batch)
        return batch