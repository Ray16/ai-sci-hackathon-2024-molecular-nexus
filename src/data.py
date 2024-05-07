import torch
from torch_geometric.data import InMemoryDataset

class GraphDataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None, pre_transform=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        # save your list of Data objects to a file using the `collate` function from `InMemoryDataset`.
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

class LoadGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        pass