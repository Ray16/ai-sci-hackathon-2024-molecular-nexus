import argparse
from src.model import GATNet, GCNModel
from src.data import GraphDataset, GraphDataModule

import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

args = argparse.ArgumentParser()

dataset = GraphDataset(root='./train_dataset/')

dm = GraphDataModule(dataset)
dm.prepare_data()

# model training
logger = CSVLogger("logs", name="AI4Science")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_channels_l1', type=int, default=16, help='Number of hidden channels after GATConv layer 1')
    parser.add_argument('--n_head_l1', type=int, default=8, help='Number of attention heads in GATConv layer 1')
    parser.add_argument('--out_channels_l2', type=int, default=16, help='Number of hidden channels after GATConv layer 2')
    parser.add_argument('--n_head_l2', type=int, default=1, help='Number of attention heads in GATConv layer 2')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs during train')
    return parser.parse_args()

args = parse_arguments()

model = GATNet(args.out_channels_l1, args.n_head_l1, args.out_channels_l2, args.n_head_l2,args.learning_rate)
checkpoint_callback = ModelCheckpoint(
    dirpath=f'./pretrained_model/{args.out_channels_l1}-{args.n_head_l1}-{args.out_channels_l2}-{args.n_head_l2}-{args.learning_rate}',
    filename='{epoch}-{step}-{val_loss:.4f}',
    monitor='val_loss',
    save_top_k=1,
    mode='min',
)
trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu",logger=logger,callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=dm)