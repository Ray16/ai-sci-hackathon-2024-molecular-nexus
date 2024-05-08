from src.model import GATNet, GCNModel
from src.data import GraphDataset, GraphDataModule

import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

dataset = GraphDataset(root='./dataset/')

dm = GraphDataModule(dataset)
dm.prepare_data()
model = GATNet()

# model training
logger = CSVLogger("logs", name="AI4Science")
checkpoint_callback = ModelCheckpoint(
    dirpath='./pretrained_model/',
    filename='{epoch}-{step}-{val_loss:.2f}',
    monitor='val_loss',
    save_top_k=3,
    mode='min',
    verbose=True
)

trainer = pl.Trainer(max_epochs=200, accelerator="gpu",logger=logger,callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=dm)