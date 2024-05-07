from src.data import GraphDataset
from torch_geometric.loader import DataLoader

# load dataset into dataloader
dataset = GraphDataset(root='./dataset/')
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=64)

for batch in loader:
    print(batch)
    print(batch[0])
    break