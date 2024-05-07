import json
import torch
import helper
from torch_geometric.data import Data

train_data = helper.load_data_from_file("data.json")

def create_pyg_graph(data):
    nodes = data.nodes
    print(nodes)
    links = data.edges
    
    node_features = [[node['atomic'], node['valence'], node['formal_charge'], 
                      int(node['aromatic']), node['hybridization'], node['radical_electrons'], 
                      node['param']['bond_type_id']] for node in nodes]
    data_x = torch.tensor(node_features, dtype=torch.float)
    print(data_x.shape)

    edge_index = [[link['source'], link['target']] for link in links]
    data_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(data_edge_index.shape)

    edge_attributes = [[link['type'], link['stereo'], int(link['aromatic']), int(link['conjugated'])] for link in links]
    data_edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    print(data_edge_attr.shape)


    target_features = [[node['param']['mass'], node['param']['charge'], node['param']['sigma'], node['param']['epsilon']] for node in nodes]
    data_y = torch.tensor(target_features, dtype=torch.float)
    print(data_y.shape)
    return Data(x=data_x, edge_index=data_edge_index, edge_attr=data_edge_attr, y=data_y)


for smiles_string in train_data:
    graph = train_data[smiles_string]
    pyg_graph = create_pyg_graph(graph)
    print(pyg_graph)
    break
