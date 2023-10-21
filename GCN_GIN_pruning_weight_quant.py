import torch
!pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

torch.manual_seed(11)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import necessary libraries
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Setup seeds for reproducibility
torch.manual_seed(11)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Install necessary dependencies
!pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html

def load_data():
    dataset = TUDataset(root='.', name='PROTEINS').shuffle()

    # Create training, validation, and test sets
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader, dataset

def plot_graphs(model, dataset, model_name):
    fig, ax = plt.subplots(4, 4)
    fig.suptitle(f'{model_name} - Graph classification')

    for i, data in enumerate(dataset[-16:]):
        out = model(data.x, data.edge_index, data.batch)
        color = "green" if out.argmax(dim=1) == data.y else "red"

        ix = np.unravel_index(i, ax.shape)
        ax[ix].axis('off')
        G = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G,
                         pos=nx.spring_layout(G, seed=0),
                         with_labels=False,
                         node_size=10,
                         node_color=color,
                         width=0.8,
                         ax=ax[ix]
                         )
    plt.show()

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

import torch.nn.utils.prune as prune

def apply_pruning_to_gcn(model, pruning_rate=0.2):
    """
    Apply magnitude pruning to the GCN model.

    Parameters:
    - model: The model to be pruned.
    - pruning_rate: Proportion of the weights to be pruned.
    """

    # Pruning 20% of the weights in conv1, conv2, and conv3 layers
    prune.l1_unstructured(model.conv1.lin, name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv2.lin, name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv3.lin, name="weight", amount=pruning_rate)

    # Optional: after pruning, the pruned weights can be removed
    # by making the pruning permanent.
    prune.remove(model.conv1.lin, 'weight')
    prune.remove(model.conv2.lin, 'weight')
    prune.remove(model.conv3.lin, 'weight')

    return model

def apply_pruning_to_gin(model, pruning_rate=0.2):
    """
    Apply magnitude pruning to the GIN model.

    Parameters:
    - model: The model to be pruned.
    - pruning_rate: Proportion of the weights to be pruned.
    """

    # Prune the first Linear layer in each GINConv layer
    prune.l1_unstructured(model.conv1.nn[0], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv2.nn[0], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv3.nn[0], name="weight", amount=pruning_rate)

    # Prune the second Linear layer in each GINConv layer
    prune.l1_unstructured(model.conv1.nn[3], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv2.nn[3], name="weight", amount=pruning_rate)
    prune.l1_unstructured(model.conv3.nn[3], name="weight", amount=pruning_rate)

    # Optional: after pruning, the pruned weights can be removed
    # by making the pruning permanent.
    prune.remove(model.conv1.nn[0], 'weight')
    prune.remove(model.conv1.nn[3], 'weight')
    prune.remove(model.conv2.nn[0], 'weight')
    prune.remove(model.conv2.nn[3], 'weight')
    prune.remove(model.conv3.nn[0], 'weight')
    prune.remove(model.conv3.nn[3], 'weight')

    return model

def quantize_weights(model, bits=8):
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1

    for param in model.parameters():
        if param.requires_grad:  # Ensure it's a trainable weight
            # Scale weights to [0, 1]
            min_val = torch.min(param)
            max_val = torch.max(param)
            param.data = (param - min_val) / (max_val - min_val)

            # Quantize
            param.data = (param * (qmax - qmin) + qmin).round()
            param.data = (param - qmin) / (qmax - qmin) * (max_val - min_val) + min_val
    return model

import time

def measure_inference_time(model, loader):
    start_time = time.time()

    with torch.no_grad():
        for data in loader:
            outputs = model(data.x, data.edge_index, data.batch)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

import os

def save_model_and_get_size(model, path):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)  # Convert bytes to megabytes
    return size

# Models, training and evaluation procedures
class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)

        return F.log_softmax(h, dim=1)

class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, dim_h):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(dataset.num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h*3, dim_h*3)
        self.lin2 = Linear(dim_h*3, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)

def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc = test(model, val_loader)

        # Print metrics every 20 epochs
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

def compare_model_size_and_time(model_name, before_size, after_size, before_time, after_time):
    print(f"{model_name} - Size Before: {before_size:.2f} MB | Size After: {after_size:.2f} MB")
    print(f"{model_name} - Inference Time Before: {before_time:.2f} s | Inference Time After: {after_time:.2f} s")
    print('-' * 80)

    print(f"{model_name} - Size Ratio: {after_size/before_size:.4f}")
    print(f"{model_name} - Inference Time Before: {after_time/before_time:.4f}")
    print('-' * 80)


if __name__ == "__main__":
    train_loader, val_loader, test_loader, dataset = load_data()

    # Dictionary to store models and their pruning functions
    models = {
        "GCN": {"model": GCN(dim_h=32), "prune": apply_pruning_to_gcn},
        "GIN": {"model": GIN(dim_h=32), "prune": apply_pruning_to_gin}
    }

    for model_name, data in models.items():
        model = data['model']

        # 1. Initial training
        model = train(model, train_loader)
        test_loss, test_acc = test(model, test_loader)
        print(f'{model_name} Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')

        # 2. Calculate and store size and inference time before pruning
        before_size = save_model_and_get_size(model, f"{model_name.lower()}_before_pruning.pt")
        before_time = measure_inference_time(model, test_loader)

        # 3. Apply pruning
        model = data['prune'](model)
        model = train(model, train_loader)  # Fine-tuning after pruning

        # 4. Calculate and store size and inference time after pruning
        after_size = save_model_and_get_size(model, f"{model_name.lower()}_after_pruning.pt")
        after_time = measure_inference_time(model, test_loader)

        # 5. Quantization
        model = quantize_weights(model)
        after_quant_time = measure_inference_time(model, test_loader)
        after_quant_size = save_model_and_get_size(model, f"{model_name.lower()}_after_quantization.pt")

        # 6. Compare and print results
        compare_model_size_and_time(f"{model_name} (Before vs After Pruning)", before_size, after_size, before_time, after_time)
        compare_model_size_and_time(f"{model_name} (After Pruning vs After Quantization)", after_size, after_quant_size, after_time, after_quant_time)

        # 7. Plot graphs
        plot_graphs(model, dataset, model_name)
