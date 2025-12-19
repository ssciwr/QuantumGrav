# Graph Neural Network models

In this section, we'll explore the architecture of the Graph Neural Network (GNN) models used in QuantumGravPy. These models are designed to learn meaningful representations from causal set graphs, which can then be used for various downstream tasks such as classification or regression.

## Structure

The GNN models in QuantumGravPy follow a modular architecture that consists of three main components:

1. **Backbone**: A sequence of GNN blocks that processes node features and graph connectivity to produce node embeddings.
2. **Optional addition**: A network for processing additional graph-level features.
3. **Frontend**: A linear block that takes the embeddings from the backbone (and optional graph features) and produces predictions for specific tasks. This can be replaced with another task specific frontend by overwriting the respective parts of `GNNModel` with your own logic.

This modular design allows for flexibility in model configuration, making it easy to adapt the architecture to different types of causal set data and analysis tasks. The separation of the backbone from the frontend enables transfer learning approaches, where a pre-trained backbone can be reused for multiple different downstream tasks:

```
Input Graph (Node features + topology)
    │
    ▼
┌──────────────┐    ┌─────────────────┐
│   Backbone   │    │  Graph Features │
│  (GNN Blocks)│    │     Network     │
└──────┬───────┘    └────────┬────────┘
       │                     │
       ▼                     ▼
┌────────────────────────────────────┐
│            Concatenate             │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│               Frontend             │
└──────────────────┬─────────────────┘
                   │
                   ▼
                 Output
```

## Backbone

The backbone consists of a sequence of `GNNBlock` modules that transform the input node features through multiple graph convolutional layers. Each GNNBlock includes:

1. A graph convolution layer (GCN, GraphConv, SageConv, etc.)
2. A batch normalization layer
3. An activation function
4. A residual connection
5. Dropout for regularization

```
    Input graph (Node features + topology)
                │              │
                ▼              │
        ┌───────────────┐      │
        │   GNN layer   │      │
        │  (GCNConv,...)│      │
        └───────┬───────┘      │
                │              │
                ▼              │
        ┌───────────────┐      │
        │   Normalize   │      │
        │ (BatchNorm..) │      │
        └───────┬───────┘      │
                │              │
                ▼              │
        ┌───────────────┐      │
        │   Normalize   │      │
        │ (BatchNorm..) │      │
        └───────┬───────┘      │
                │              │
                ▼              │
        ┌────────────────┐     │
        │   Activation   │     │
        │  (ReLu...)     │     │
        └───────┬────────┘     │
                │              │
                ▼              ▼
             ┌─────────────────────┐
             │          +          │  residual connection
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │      Dropout        │  only active during training
             └──────────┬──────────┘
                        │
                        ▼
                      Output
```

This implementation follows modern deep learning practices, with residual connections helping to mitigate the vanishing gradient problem and enabling the training of deeper networks. Each block is set up with a set of parameters:

```python
# Example GNNBlock configuration
{
    "in_dim": 64,
    "out_dim": 64,
    "dropout": 0.3,
    "gnn_layer_type": "GCNConv",
    "normalizer": "BatchNorm",
    "activation": "ReLU",
    "gnn_layer_args": [arg1, arg2, ...]
    "gnn_layer_kwargs":{
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
    "norm_args": [arg1, arg2, ...]
    "norm_kwargs": {
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
    "activation_args": [arg1, arg2, ...]
    "activation_kwargs": {
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
}
```

The backbone processes the node features and graph connectivity through these blocks sequentially, and the final node embeddings are pooled (using a configurable pooling operation such as mean, sum, or max pooling) to obtain a graph-level representation. It makes sense to familiarize yourself with the `pytorch_geometric` documentation if you don't have already, to learn how the GNN layers, activation functions etc work.

The `gnn_layer_type`, `activation` and `normalizer` can be chosen from a set of predefined layers:
```python
gnn_layers: dict[str, torch.nn.Module] = {
    "gcn": tgnn.conv.GCNConv,
    "gat": tgnn.conv.GATConv,
    "sage": tgnn.conv.SAGEConv,
    "gco": tgnn.conv.GraphConv,
}

normalizer_layers: dict[str, torch.nn.Module] = {
    "identity": torch.nn.Identity,
    "batch_norm": torch.nn.BatchNorm1d,
    "layer_norm": torch.nn.LayerNorm,
}

activation_layers: dict[str, torch.nn.Module] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}


pooling_layers: dict[str, torch.nn.Module] = {
    "mean": tgnn.global_mean_pool,
    "max": tgnn.global_max_pool,
    "sum": tgnn.global_add_pool,
}
```
You can add new layers by using the supplied `register_` functions in the `utils` module. Consult the [API documentation](./api.md) for more details.

## Optional graph-level features

In many real-world scenarios, we have additional graph-level features that can't be naturally represented as node features. The `GraphFeaturesBlock` allows for processing these auxiliary features and integrating them with the learned graph representation from the backbone.

The `GraphFeaturesBlock` is a sequence of linear layers with activation functions that processes these global graph features. The outputs of the GNN backbone and the graph features network are concatenated before being passed to the frontend classifier. It's construction parameters are similar to the `GNNBlock`:

```python
# Example GraphFeaturesBlock configuration
{
    "input_dim": 10,
    "output_dim": 32,
    "hidden_dims": [64, 64],
    "activation": "ReLU",
    "norm_args": [arg1, arg2, ...]
    "norm_kwargs": {
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
    "activation_args": [arg1, arg2, ...]
    "activation_kwargs": {
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
}
```

This approach allows the model to incorporate both local (node-level) and global (graph-level) information when making predictions. By customizing the `hidden_dims` you can make the network shallower or deeper or wider or narrower.


Edges features are currently not supported out of the box.

## Frontends

The frontend model takes the combined embeddings from the backbone and optional graph features network and produces predictions for specific tasks. The `ClassifierBlock` is based on the `LinearSequential` module, which supports multi-objective classification by allowing multiple output layers, each corresponding to a different task. It's construction parameters are once more similar to the `GNNBlock` and `GraphFeaturesBlock` models, e.g.:

```python
# Example ClassifierBlock configuration
{
    "input_dim": 96,  # Combined dimension from backbone and graph features
    "output_dims": [2, 3],  # Two tasks: binary classification and 3-class classification
    "hidden_dims": [128, 64],
    "activation": "ReLU",
    "backbone_kwargs":
        - {
            key1: kwarg1,
            key2: kwarg2,
            key3: ...
        }
        - {
            key1: kwarg1,
            key2: kwarg2,
            key3: ...
        }
       - {
            key1: kwarg1,
            key2: kwarg2,
            key3: ...
        }
    output_kwargs:
        - {
            key1: kwarg1,
            key2: kwarg2,
            key3: ...
        },
        - {
            key1: kwarg1,
            key2: kwarg2,
            key3: ...
        }
    "activation_args": [arg1, arg2, ...]
    "activation_kwargs": {
        key1: kwarg1,
        key2: kwarg2,
        key3: ...
    }
}
```
By customizing the `hidden_dims` you can make the network shallower or deeper or wider or narrower.
In this way, transfer learning is supported - we can change the frontend depending on the task, while keeping the model that produces embeddings.
The output layers are implemneted as one linear layer per task, and each of them can be given a list of kwargs. The same holds for the input- and hidden layers: for each of them, the `backbone_kwargs` config node can contain a set of kwargs, too. you can leave out some of them if desired, but for skipping one you need to put in an empty dictionary because the code maps kwargs to layers by index in the list.


## Configuration-driven model creation

All model components can be created from configuration dictionaries, making it easy to experiment with different architectures without changing the code. The `from_config` class methods in each component allow for declarative model definition through YAML configuration files.

```python
# Example full model configuration
{
    "encoder": [
        {
            "in_dim": 8,
            "out_dim": 64,
            "gnn_layer_type": "GCNConv",
            "normalizer": "BatchNorm",
            "activation": "ReLU"
        },
        {
            "in_dim": 64,
            "out_dim": 64,
            "gnn_layer_type": "GCNConv",
            "normalizer": "BatchNorm",
            "activation": "ReLU"
        }
    ],
    "pooling_layer": "mean",
    "graph_features_net": {
        "input_dim": 10,
        "output_dim": 32,
        "hidden_dims": [64, 64],
        "activation": "ReLU"
    },
    "classifier": {
        "input_dim": 96,  # 64 from backbone + 32 from graph features
        "output_dims": [2],
        "hidden_dims": [128, 64],
        "activation": "ReLU"
    }
}
```

This configuration-based approach aligns with the overall philosophy of QuantumGravPy, which emphasizes a separation between configuration and code to facilitate rapid experimentation and reproducibility. Note how the number of blocks in the `encoder` part of the config determines the architecture of the backbone model - each block will create a `GNNBlock` instance, and the data is processed through them sequentially. In the same way, the number of `hidden_dims` will make the linear models deeper or shallower, and the supplied numbers determine the model width.
The output dim in `classifier` determines the number of classification tasks.

Note that right now, only the `ClassifierBlock` is explicitly supported.

## Transfer learning and model reuse

The separation of backbone and frontend enables effective transfer learning strategies. A backbone trained on a large dataset of causal sets can capture general properties of causal set graphs, which can then be reused for multiple downstream tasks by attaching different frontend classifiers.
Currently, only the classifier block is supproted, but later on, we plan to generalize this such that the system can accomodate generative models as well.

To extract the embeddings from a trained model for reuse, you can use the `get_embeddings` method of the `GNNModel` class, which returns the output of the backbone after pooling, ready to be used for other tasks.

## Full config example:
A full configuration that defines a model as it would appear in a YAML config file  could read:
```yaml
  encoder:
    - in_dim: 12
      out_dim: 128
      dropout: 0.3
      gnn_layer_type: "sage"
      normalizer: "batch_norm"
      activation: "relu"
      norm_args:
        - 128
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
    - in_dim: 128
      out_dim: 256
      dropout: 0.3
      gnn_layer_type: "sage"
      normalizer: "batch_norm"
      activation: "relu"
      norm_args:
        - 256
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
    - in_dim: 256
      out_dim: 128
      dropout: 0.3
      gnn_layer_type: "sage"
      normalizer: "batch_norm"
      activation: "relu"
      norm_args:
        - 128
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
  pooling_layer: mean
  classifier:
    input_dim: 128
    output_dims:
      - 2
    hidden_dims:
      - 48
      - 18
    activation: "relu"
    backbone_kwargs: [{}, {}]
    output_kwargs: [{}]
    activation_kwargs: [{ "inplace": False }]
```

Here, we use a chain of three `GraphSage`-based `GNNBlocks`, using batch normalization, a dropout probability of 0.3 and a `ReLU` activation function, followed by a global `mean` pooling layer and a classifer that has an input layer of size 128, two hidden layers of sizes 48 and 18, and an output layer of size 2, i.e., 2 classification tasks. We also make add some arguments for the constructor of the activation (`activation_kwargs`). The first `GNNBlock` must have an input dimension that corresponds to the dimensionality of the node features per node. We have no `GraphfeaturesBlock` in this case, but if we had it would look similar to the `classifier` block, the config node would just be called `graph_features_net` instead of `classifier`.

Armed with this knowledge, we now can proceed to model training.
