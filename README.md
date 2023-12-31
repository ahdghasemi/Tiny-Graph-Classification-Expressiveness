# Graph-Classification-Expressiveness

This is the first part of my exploration to decrease GNN model sizes in different applications. In the first step, we first employed both pruning and weight quantization in  graph classification expressiveness. The model size decreases around 3 ~ 5%. In the next step, we deploy dynamic post-training quantization.

This repository contains two codes for graph classification using PyTorch Geometric, a library designed for handling graph-based deep learning tasks. each of these codes include two models: Graph Convolutional Networks (GCN) and Graph Isomorphism Networks (GIN). The code is written is PyTorch (PyTorch Geometric). The primary goal is to demonstrate the process of minimzing the size of trained model to be used on-devices such as phones and microprocessors. The used techniques are pruning, weight quantization, and dynamic post-training quantization. Both model sizes in MB and inference time in second are compared before and after applying techniques. 

The code "GNN_PyTorch_Geometric_Effective_Quantization_Final.py" includes deploying pruning and weight quantization. In addition, the code "GNN_PyTorch_Geometric_Effective_Quantization_Final.py" include applying dynamic post-training quantization to the original code.


### Getting Started

#### Prerequisites

Before running the code, ensure you have the required libraries installed by running the following command:

```python
import torch
!pip install -q torch-scatter~=2.1.0 torch-sparse~=0.6.16 torch-cluster~=1.6.0 torch-spline-conv~=1.2.1 torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```

#### Dataset
This code uses the "PROTEINS" dataset from the TUDataset, which is a collection of graph datasets. The dataset is divided into training, validation, and test sets.

#### Models
1. **GCN (Graph Convolutional Networks):** The GCN model consists of three GCN layers followed by a linear classifier. It utilizes the GCNConv layer for graph convolutions.

2. **GIN (Graph Isomorphism Networks):** The GIN model consists of three GINConv layers, each comprising two linear layers, batch normalization, and ReLU activation. It uses global node aggregation to produce graph-level embeddings.

#### Training
The training process involves the following steps:

1. Model initialization.
2. Training for a specified number of epochs.
3. Calculating training and validation loss and accuracy.
4. Optional fine-tuning after pruning.
5. Quantization of model weights.


#### Pruning
Pruning is performed to reduce the model size by removing unimportant weights. Two types of pruning are applied, depending on the model:

#### GCN Pruning
In the GCN model, 20% of the weights in each GCN layer are pruned using L1 unstructured magnitude pruning. Pruned weights can be removed optionally to make pruning permanent.

#### GIN Pruning
In the GIN model, 20% of the weights in both linear layers within each GINConv layer are pruned using L1 unstructured magnitude pruning. Pruned weights can be removed optionally.

#### Quantization
Quantization is applied to the model weights to reduce their precision, further reducing model size. The model weights are quantized to 8 bits.

#### Model Size and Inference Time Comparison
The code allows you to compare the model size and inference time before and after pruning and after quantization. This comparison helps understand the trade-offs between model size reduction and inference speed.

### Dynamic Post-Training Quantization

In our quest to optimize GNN models for resource-constrained devices, we've introduced dynamic post-training quantization as a powerful technique. This step is implemented in the code titled **GNN_PyTorch_Geometric_Dynamic_Quantization_Final.py**.

#### What is Dynamic Post-Training Quantization?

Dynamic post-training quantization is a method for reducing the precision of model weights, making them more suitable for deployment on devices with limited computational resources. Unlike static quantization, which uses a fixed set of quantization ranges, dynamic quantization dynamically determines these ranges during inference based on the actual distribution of weights. This results in a more efficient model with minimal loss of accuracy.

#### How it Works

In the code, dynamic post-training quantization is applied to the GNN models after training and optionally after pruning. Here's how it works:

1. After the model is trained and optionally pruned, we apply dynamic post-training quantization to the model's weights.

2. During inference, the model uses the quantized weights, which reduces the memory and computational requirements.

3. Dynamic post-training quantization adjusts the quantization ranges based on the weights' distribution, ensuring that the model maintains high accuracy while being more efficient.

By integrating dynamic post-training quantization into our optimization pipeline, we further reduce the model's size and inference time while preserving the quality of graph classification.

Feel free to explore the **GNN_PyTorch_Geometric_Dynamic_Quantization_Final.py** code to see how dynamic post-training quantization is applied and experiment with different configurations to optimize your GNN models for deployment on devices with limited resources.


#### Graph Visualization
The code also includes a function to visualize graph classifications for both GCN and GIN models. It shows how well the models are performing on a subset of the dataset.

#### Usage
To execute the code, run it in your preferred Python environment. The repository provides two main models, GCN and GIN, and their respective training, pruning, and quantization steps. Make sure to specify the number of epochs, training data, and other hyperparameters based on your specific use case.
