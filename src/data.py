import torch
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
        self.train_dataset, self.val_dataset = random_split(dataset,[0.8,0.2])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,pin_memory=True)