import os
import torch
from src.data import GraphDataset
from src.utils import load_json_data
from multiprocessing import Pool
from torch_geometric.data import Data
from src.model import GATNet, GCNModel
from src.data import GraphDataset, GraphDataModule

import pytorch_lightning as pl

NCPUS = int(0.9*os.cpu_count())

all_data = load_json_data('validation_example.json')

def create_pyg_graph(index):
    data = list(all_data.values())[index]
    nodes = data['nodes']
    links = data['links']
    
    node_features = [[node['atomic'], node['valence'], node['formal_charge'], 
                      int(node['aromatic']), node['hybridization'], node['radical_electrons'], 
                      node['id']] for node in nodes]
    data_x = torch.tensor(node_features, dtype=torch.float)
    #print(data_x.shape)

    edge_index = [[link['source'], link['target']] for link in links]
    data_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #print(data_edge_index.shape)

    edge_attributes = [[link['type'], link['stereo'], int(link['aromatic']), int(link['conjugated'])] for link in links]
    data_edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    #print(data_edge_attr.shape)

    #target_features = [[node['param']['mass'], node['param']['charge'], node['param']['sigma'], node['param']['epsilon']] for node in nodes]
    #data_y = torch.tensor(target_features, dtype=torch.float)
    #print(data_y.shape)
    #return Data(x=data_x, edge_index=data_edge_index, edge_attr=data_edge_attr, y=data_y)
    return Data(x=data_x, edge_index=data_edge_index, edge_attr=data_edge_attr)

def process_data_parallel(num_data):
    with Pool(processes=NCPUS) as pool:
        data_list = pool.map(create_pyg_graph, range(num_data))
    return data_list

data_list = process_data_parallel(len(all_data))

# Save dataset to disk
root_dir = './validation_dataset'
dataset = GraphDataset(root=root_dir, data_list=data_list)

dm = GraphDataModule(dataset)
dm.prepare_data()

trainer = pl.Trainer(accelerator="gpu")

# load pre-trained model
pretrained_model_path = "./pretrained_model/epoch=199-step=15000-val_loss=0.17.ckpt"
new_model = GATNet.load_from_checkpoint(checkpoint_path=pretrained_model_path)
# make 
predictions = trainer.predict(new_model, datamodule=dm)
# predicted epsilon/concatenate all molecules
predictions_tensor = torch.cat(predictions)
print(predictions_tensor)