import pandas as pd
from glob import glob

file_list = []
loss = []
for folder in glob('./archive/pretrained_model/*/'):
    files = glob(folder+'*.ckpt')
    file_list+=files
loss = [float(file.split('val_loss=')[1].split('.ckpt')[0]) for file in file_list]
hyper_parameter = [file.split('/')[3] for file in file_list]
df = pd.DataFrame({'hp':hyper_parameter,'val_loss':loss})
df = df.sort_values('val_loss',ascending=True)
df.to_csv('hyperparameter_tuning_result.csv',index=False)