import torch
import pytorch_lightning as pl
from torch_geometric.nn import NNConv, GATConv, CGConv
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU

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

class CGConvNet(pl.LightningModule):
    def __init__(self, out_channels_l1, learning_rate):
        super().__init__()
        self.cgconv1 = CGConv(7, 4)
        self.cgconv2 = CGConv(7, 4)
        self.cgconv3 = CGConv(7, 4)
        self.learning_rate = learning_rate
        self.out_channels_l1 = out_channels_l1

        self.edge_network = Sequential(
            Linear(4, self.out_channels_l1),
            ReLU(),
            Linear(self.out_channels_l1, 28)  
        )

        self.nnconv = NNConv(7, 4, self.edge_network, aggr='mean')
        self.fc = Linear(4, 4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.cgconv1(x, edge_index, edge_attr))
        x = F.relu(self.cgconv2(x, edge_index, edge_attr))
        x = F.relu(self.cgconv3(x, edge_index, edge_attr))
        x = F.relu(self.nnconv(x, edge_index, edge_attr))
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