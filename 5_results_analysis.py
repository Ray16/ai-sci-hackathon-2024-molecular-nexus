import os
import json
import numpy as np
import helper as helper
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import CGConvNet

import pytorch_lightning as pl

dataloader = torch.load('dataloader/val_dataloader.pth')
data_x = torch.cat([mol.x for mol in list(dataloader.dataset)])
data_y = torch.cat([mol.y for mol in list(dataloader.dataset)])
print(data_y.shape)

trainer = pl.Trainer(accelerator="gpu")
# load pre-trained model
pretrained_model_path = "pretrained_model_CGConvNet/128-0.005/epoch=508-step=38175-val_loss=0.0046.ckpt"
new_model = CGConvNet.load_from_checkpoint(checkpoint_path=pretrained_model_path, out_channels_l1=128, learning_rate=0.005)
predictions = trainer.predict(new_model, dataloader)
predictions = torch.cat(predictions)
print(predictions.shape)

mae = torch.abs(predictions - data_y)

# histogram for mae distributions of 4 attrs

attrs = ['mass', 'charge', 'sigma', 'epsilon']

for i in range(4):
    plt.figure(figsize=(4,3))
    plt.hist(mae[:, i].numpy())
    plt.xlabel('MAE')
    plt.ylabel('# Nodes')
    plt.yscale('log')
    plt.title(attrs[i])
    plt.savefig(f'./plots/hist_{attrs[i]}', dpi=300, bbox_inches='tight')

# Per element mae

atomic_dict = {}

for i in range(len(data_x)):
    atomic = data_x[i][0].numpy().item()
    if atomic not in atomic_dict:
        atomic_dict[atomic] = [int(i)]
    else:
        atomic_dict[atomic].append(int(i))

atomic_name_dict = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    11: 'Na',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I'
}
atomic_list = [1,6,7,8,9,11,16,17,35,53]

atomic_mat = []
for atomic in atomic_list:
    
    indices = np.array(atomic_dict[atomic])
    atomic_mean = torch.mean(mae[indices], dim=0)
    atomic_mat.append(atomic_mean)
    atomic_std = torch.std(mae[indices], dim=0)
    
    plt.figure(figsize=(4,3))
    plt.bar(range(4), atomic_mean, yerr=atomic_std, capsize=3)
    plt.xticks(ticks=range(4), labels=attrs)
    plt.xlabel('Force Field Parameters')
    plt.ylabel('MAE')
    plt.title(f'MAE For {atomic_name_dict[atomic]} Nodes (# Nodes = {len(indices)})')
    plt.savefig(f'./plots/atomic_{atomic_name_dict[atomic]}', dpi=300, bbox_inches='tight')


# Colormap

plt.figure(figsize=(4,3))
sns.heatmap(atomic_mat, cmap='coolwarm')
plt.xticks(ticks=[i+0.5 for i in range(4)], labels=attrs)
plt.yticks(ticks=[i+0.5 for i in range(10)], labels=[atomic_name_dict[i] for i in atomic_list])
plt.title('MAE By Elements')
plt.savefig('./plots/atomic_all', dpi=300, bbox_inches='tight')
