import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 4

hp_result = pd.read_csv('./hyperparameter_tuning_result.csv')
hp_result = hp_result.sort_values('val_loss',ascending=True)
hp_result = hp_result.reset_index(drop=True)

hp_result_new = pd.DataFrame({'hp':list(hp_result.hp),'val_loss':list(hp_result.val_loss)})
plt.figure(figsize=(10,3))
plt.bar(hp_result_new.hp, hp_result_new.val_loss,color='#800000')
plt.xlabel('combination of hyperparameters')
plt.ylabel('val_loss')
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig('./plots/hp_bar.jpg',dpi=300,bbox_inches='tight')
plt.close()

# list_hp = ['out_channels_l1','n_head_l1','out_channels_l2','n_head_l2','learning_rate']

# for idx, hp in enumerate(list_hp):
#     hp_result[hp] = [float(ele.split('-')[idx]) for ele in hp_result['hp']]

# plt.rcParams['xtick.labelsize'] = 10
# # effect of single hyperparameter on model performance
# for hp in list_hp:
#     avg_loss = [] 
#     for unique_hp in hp_result[hp].unique():
#         avg_loss.append(np.average(hp_result[hp_result[hp]==unique_hp].val_loss))
#     plt.bar(hp_result[hp].unique(),avg_loss)
#     plt.xlabel(hp)
#     plt.ylabel('val_loss')
#     plt.tight_layout()
#     plt.savefig(f'plots/{hp}.jpg',dpi=300,bbox_inches='tight')
#     plt.close()
