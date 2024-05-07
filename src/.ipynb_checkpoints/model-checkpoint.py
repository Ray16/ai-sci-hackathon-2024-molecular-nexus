import torch
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(7, 16)  # Input features: 7, Output features: 16 
        self.conv2 = GATConv(16, 32)  # Using edge_attr might require increasing input dimension
        self.out = torch.nn.Linear(32, 4)  # Map to final 4 output values

    def forward(self, data):
        #x, edge_index = data.x, data.edge_index
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index,edge_attr)
        x = torch.relu(x) 
        print(f'x_shape:{x.shape}')
        #print(f'cat_shape:{torch.cat([x, edge_attr], dim=1).shape}')
        print(f'edge_index_shape:{edge_index.shape}')
        # If using edge attributes, you might concatenate in the next layer
        #x = self.conv2(torch.cat([x, edge_attr], dim=1), edge_index)
        x = self.conv2(x, edge_index,edge_attr)

        x = torch.relu(x) 

        x = self.out(x)
        return x