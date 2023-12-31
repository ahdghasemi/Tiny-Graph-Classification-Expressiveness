import torch
!pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

torch.manual_seed(11)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.quantization as quantization
import torch.nn.utils.prune as prune
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Setup seeds for reproducibility
torch.manual_seed(11)

import os

def get_model_file_size(model_path):
    """
    Returns the size of the model file in MB.
    """
    size_bytes = os.path.getsize(model_path)
    size_megabytes = size_bytes / (1024 * 1024)  # Convert bytes to megabytes
    return size_megabytes

def calculate_percentage_decrease(original_size, new_size):
    """
    Returns the percentage decrease from original_size to new_size.
    """
    return ((original_size - new_size) / original_size) * 100

def quantize_model(model):
    """
    Applies quantization to the given model and returns the quantized version.
    """
    model.eval()
    model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    # Here, you'd calibrate the model using a calibration dataset, but we're skipping this for simplicity.
    torch.quantization.convert(model, inplace=True)
    return model

def load_data():
    dataset = TUDataset(root='.', name='PROTEINS').shuffle()
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader, dataset

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

class GCN(torch.nn.Module):
    def __init__(self, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        hG = global_mean_pool(h, batch)
        h = F.dropout(hG, p=0.25, training=self.training)
        h = self.lin(h)
        return F.log_softmax(h, dim=1)

class GIN(torch.nn.Module):
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(Linear(dataset.num_node_features, dim_h), ReLU()))
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = global_add_pool(h, batch)
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin(h)
        return F.log_softmax(h, dim=1)

def train(model, loader, val_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)  # L1 regularization added
    epochs = 50  # Reduced epochs for faster training

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = test(model, val_loader)
        if(epoch % 20 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')
    return model

@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
    return loss, acc


if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset = load_data()

    # Train and test GCN model
    gcn = GCN(dim_h=32)
    gcn = train(gcn, train_loader, val_loader)
    test_loss, test_acc = test(gcn, test_loader)
    print(f'Original GCN Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

    # For GCN
    torch.save(gcn.state_dict(), "original_gcn_model.pth")
    original_gcn_size = get_model_file_size("original_gcn_model.pth")

    # Quantize and Test GCN model
    gcn_quantized = quantize_model(gcn)
    torch.save(gcn_quantized.state_dict(), "quantized_gcn_model.pth")
    quantized_gcn_size = get_model_file_size("quantized_gcn_model.pth")

    print(f"Original GCN model size: {original_gcn_size:.2f} MB")
    print(f"Quantized GCN model size: {quantized_gcn_size:.2f} MB")
    # print(f"GCN model size ratio: {((original_gcn_size - quantized_gcn_size)/original_gcn_size)*100:.2f} MB")
    print(f"GCN model size ratio (original/quantized): {((original_gcn_size/quantized_gcn_size))*100:.2f}X ")


    # Train and test GIN model
    gin = GIN(dim_h=32)
    gin = train(gin, train_loader, val_loader)
    test_loss, test_acc = test(gin, test_loader)
    print(f'Original GIN Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

    # For GIN
    torch.save(gin.state_dict(), "original_gin_model.pth")
    original_gin_size = get_model_file_size("original_gin_model.pth")

    # Quantize and Test GIN model
    gin_quantized = quantize_model(gin)
    torch.save(gin_quantized.state_dict(), "quantized_gin_model.pth")
    quantized_gin_size = get_model_file_size("quantized_gin_model.pth")

    print(f"Original GIN model size: {original_gin_size:.2f} MB")
    print(f"Quantized GIN model size: {quantized_gin_size:.2f} MB")
    # print(f"GIN model size ratio: {((original_gin_size - quantized_gin_size)/original_gin_size)*100:.2f}X ")
    # print(f"GIN model size ratio: {((quantized_gin_size)/original_gin_size)*100:.2f}X ")
    print(f"GIN model size ratio (original/quantized): {((original_gin_size/quantized_gin_size))*100:.2f}X ")

    # Save the quantized model for on-device deployment
    torch.save(gin_quantized.state_dict(), "quantized_gin_model.pth")
