import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hp_result = pd.read_csv('hyperparameter_tuning_result.csv')

list_hp = ['out_channels_l1','n_head_l1','out_channels_l2','n_head_l2','learning_rate']

for idx, hp in enumerate(list_hp):
    hp_result[hp] = [float(ele.split('-')[idx]) for ele in hp_result['hp']]

# effect of single hyperparameter on model performance
for hp in list_hp:
    avg_loss = []
    for unique_hp in hp_result[hp].unique():
        avg_loss.append(np.average(hp_result[hp_result[hp]==unique_hp].val_loss))
    plt.plot(hp_result[hp].unique(),avg_loss)
    plt.xlabel(hp)
    plt.ylabel('val_loss')
    plt.tight_layout()
    plt.savefig(f'plots/{hp}.jpg',dpi=300,bbox_inches='tight')
    plt.close()