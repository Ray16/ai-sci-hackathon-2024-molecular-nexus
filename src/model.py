import torch
import pytorch_lightning as pl
from torch_geometric.nn import GATConv

class GATNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(7, 16)
        self.conv2 = GATConv(16, 32)
        self.out = torch.nn.Linear(32, 4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index,edge_attr)
        x = torch.relu(x) 
        x = self.conv2(x, edge_index,edge_attr)
        x = torch.relu(x) 
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = torch.nn.functional.mse_loss(out, batch.y.float())
        self.log("train_loss", loss, batch_size = 32, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = torch.nn.functional.mse_loss(out, batch.y.float())
        self.log("val_loss", loss, batch_size = 32, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        return optimizer