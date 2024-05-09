import torch
import pytorch_lightning as pl
from torch_geometric.nn import NNConv, GATConv, CGConv
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU

from torch_geometric.nn import CuGraphGATConv, FusedGATConv, GATv2Conv, TransformerConv, AGNNConv, TAGConv

class GATNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
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

class CuGraphGATNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = CuGraphGATConv(7, self.out_channels_l1, heads=self.n_head_l1, concat=False)
        self.conv2 = CuGraphGATConv(self.out_channels_l1, self.out_channels_l2, heads=self.n_head_l2, concat=False)
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
    

class FusedGATNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = FusedGATConv(7, self.out_channels_l1, heads=self.n_head_l1, concat=False)
        self.conv2 = FusedGATConv(self.out_channels_l1, self.out_channels_l2, heads=self.n_head_l2, concat=False)
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
    

class GATv2Net(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = GATv2Conv(7, self.out_channels_l1, heads=self.n_head_l1, concat=False)
        self.conv2 = GATv2Conv(self.out_channels_l1, self.out_channels_l2, heads=self.n_head_l2, concat=False)
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


class TransformerNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = TransformerConv(7, self.out_channels_l1, heads=self.n_head_l1, concat=False)
        self.conv2 = TransformerConv(self.out_channels_l1, self.out_channels_l2, heads=self.n_head_l2, concat=False)
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
    

class AGNNNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = AGNNConv(7, self.out_channels_l1)
        self.conv2 = AGNNConv(self.out_channels_l1, self.out_channels_l2)
        self.fc = torch.nn.Linear(self.out_channels_l2, 4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = torch.relu(x) 
        x = self.conv2(x, edge_index)
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
    

class TAGNet(pl.LightningModule):
    def __init__(
            self,
            out_channels_l1=32,
            n_head_l1=16,
            out_channels_l2=64,
            n_head_l2=8,
            learning_rate=0.005
        ):
        super().__init__()
        self.out_channels_l1 = out_channels_l1
        self.n_head_l1 = n_head_l1
        self.out_channels_l2 = out_channels_l2
        self.n_head_l2 = n_head_l2
        self.learning_rate = learning_rate
        self.conv1 = TAGConv(7, self.out_channels_l1)
        self.conv2 = TAGConv(self.out_channels_l1, self.out_channels_l2)
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


