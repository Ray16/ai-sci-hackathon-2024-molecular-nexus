from src.model import GATNet
from src.data import GraphDataset, GraphDataModule
import pytorch_lightning as pl


wandb_logger = WandbLogger(project="AI4Science_hackathon")
model = GATNet()
dataset = GraphDataset(root='./dataset/')

dm = GraphDataModule(dataset)
trainer = pl.Trainer(max_epochs=200, accelerator="auto")
trainer.fit(model, datamodule=dm)