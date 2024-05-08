
# load pre-trained model
new_model = GATNet.load_from_checkpoint(checkpoint_path="./pretrained_model/epoch=199-step=15000-val_loss=0.17.ckpt")

'''
# make predictions
predictions = trainer.predict(new_model, datamodule=dm)
predictions_tensor = torch.cat(predictions)
print(predictions_tensor)
print(predictions_tensor.shape)
# load ground truth data
test_dataloader = torch.load('./dataloader/test_dataloader.pth')
true_y = []
for data in test_dataloader:
    true_y.append(data.y)
true_y_tensor = torch.cat(true_y)
print(true_y_tensor)
print(true_y_tensor.shape)
'''


### old - normalization
'''
norm_stat = pd.read_csv('norm_stat.csv')

def inv_norm(mol_ffp):
    print(mol_ffp)
    print(torch.tensor(norm_stat['std']))
    inv_norm_mol_ffp = mol_ffp * torch.tensor(norm_stat['std']) + torch.tensor(norm_stat['mean'])
    return inv_norm_mol_ffp
# inverse normalization
inv_norm_ffp = [inv_norm(mol_ffp) for mol_ffp in predictions]
df_predictions = pd.DataFrame(torch.cat(predictions))
df_predictions.to_csv('predictions.csv',index=False)
df_inv_norm_ffp = pd.DataFrame(torch.cat(inv_norm_ffp))
df_inv_norm_ffp.to_csv('inv_norm_predictions.csv',index=False)
'''