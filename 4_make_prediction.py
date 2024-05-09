import os
import json
import pandas as pd
import numpy as np
import helper as helper
import torch
from src.data import PredGraphDataModule
from final_evaluation_modified import add_data_from_prediction
from src.utils import load_json_data
from multiprocessing import Pool
from torch_geometric.data import Data
from src.model import CGConvNet
from src.data import GraphDataset

import pytorch_lightning as pl

NCPUS = int(0.9*os.cpu_count())

list_validation_data = ['validation_masked.json','solvent_masked.json','permutation_masked.json','josh_masked.json']

for validation_data in list_validation_data:
    print(f'making predictions for {validation_data}')
    input_json_dir = 'final_evaluation'
    output_dir = 'final_evaluation_predictions'
    os.makedirs(output_dir,exist_ok=True)
    print(os.path.join(input_json_dir,validation_data))
    all_data = load_json_data(os.path.join(input_json_dir,validation_data))

    def create_pyg_graph(index):
        data = list(all_data.values())[index]
        nodes = data['nodes']
        links = data['links']
        
        node_features = [[node['atomic'], node['valence'], node['formal_charge'], 
                        int(node['aromatic']), node['hybridization'], node['radical_electrons'], 
                        node['id']] for node in nodes]
        data_x = torch.tensor(node_features, dtype=torch.float)

        edge_index = [[link['source'], link['target']] for link in links]
        data_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        edge_attributes = [[link['type'], link['stereo'], int(link['aromatic']), int(link['conjugated'])] for link in links]
        data_edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        return Data(x=data_x, edge_index=data_edge_index, edge_attr=data_edge_attr)

    def process_data_parallel(num_data):
        with Pool(processes=NCPUS) as pool:
            data_list = pool.map(create_pyg_graph, range(num_data))
        return data_list

    data_list = process_data_parallel(len(all_data))

    # Save dataset to disk
    root_dir = f'./validation_dataset/{validation_data}'
    dataset = GraphDataset(root=root_dir, data_list=data_list)

    dm = PredGraphDataModule(dataset,batch_size=1)

    trainer = pl.Trainer(accelerator="gpu")

    # load pre-trained model
    pretrained_model_path = "pretrained_model_CGConvNet/128-0.005/epoch=508-step=38175-val_loss=0.0046.ckpt"
    new_model = CGConvNet.load_from_checkpoint(checkpoint_path=pretrained_model_path, out_channels_l1=128, learning_rate=0.005)
    # make predictions
    predictions = trainer.predict(new_model, datamodule=dm)

    # make write predictions to json file 
    result_dict = add_data_from_prediction(helper.load_data_from_file(os.path.join(input_json_dir,validation_data)),os.path.join(output_dir,validation_data),predictions)