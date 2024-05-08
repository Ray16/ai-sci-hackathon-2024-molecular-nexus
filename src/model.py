import torch
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATConv

class GATNet(pl.LightningModule):
    def __init__(self,out_channels_l1,n_head_l1,out_channels_l2,n_head_l2,learning_rate):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = GATConv(7, self.out_channels_l1, heads=self.n_head_l1, concat=False)
        self.conv2 = GATConv(self.out_channels_l1, self.out_channels_l2, heads=self.n_head_l2, concat=False)
        self.fc = torch.nn.Linear(self.out_channels_l2, 4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index,edge_attr)
        x = torch.relu(x) 
        x = self.conv2(x, edge_index,edge_attr)
        x = torch.relu(x)
        x = self.fc(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        return optimizer
    

class GCNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(7, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc = torch.nn.Linear(32, 4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index,edge_attr)
        x = torch.relu(x) 
        x = self.conv2(x, edge_index,edge_attr)
        x = torch.relu(x)
        x = self.fc(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=5e-4)
        return optimizer