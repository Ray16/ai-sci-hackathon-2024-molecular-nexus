from src.model import GATNet
from src.data import GraphDataset

from torch_geometric.loader import DataLoader

# load dataset into dataloader
dataset = GraphDataset(root='./dataset/')
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=64)

for batch in loader:
    model = GATNet()
    output = model(batch[0]) 
    print(output)
    print(output.shape)  # [num_atoms, 4]
    break