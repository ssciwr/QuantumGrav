# Hyperparameter Optimization with Optuna

The `QuantumGrav` Python package lets users customize hyperparameters when building, training, validating, and testing GNN models. Choosing the right values is crucial for model performance.

To accelerate this process, we developed QGTune, a subpackage that uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html) to automatically find optimal hyperparameters for specific objectives (e.g. minimizing loss or maximizing accuracy).

## Define Optuna search space

To use Optuna, we first need to define the hyperparameter search space with methods from [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html), including:

* `suggest_categorical()`: suggest a value for the categorical parameter
* `suggest_float()`: suggest a value for the floating point parameter
* `suggest_int()`: suggest a value for the integer parameter

Normally, users can call Optuna's suggestion functions directly when creating, training, validating, or testing their models (see [Optuna's code examples](https://optuna.org/#code_examples)). However, since the `QuantumGrav` package allows these steps to be configured via a YAML file, it is more convenient to handle hyperparameter tuning the same way.

To define search space in `QuantumGrav`, users need three setting files:

* Base setting file: contains all configurations for using the `QuantumGrav` Python package (see [the configuration `dict`](./training_a_model.md#the-configuration-dict))
* Search space file: specifies the hyperparameters to optimize and their ranges
* Dependency mapping file: defines dependencies between hyperparameters. A common case is in GNN layers, where the input dimension of one layer must match the output dimension of the previous layer.

### Base setting vs. Search space

The search space file follows the same structure as the base setting file but replaces hyperparameter values with their ranges. For example:

```yaml
model:
  name: "QuantumGravBase"
  gcn_net:
    - in_dim: 12
      out_dim: 128
      dropout: 0.3
      gnn_layer_type: "sage"
      normalizer: "batch_norm"
      activation: "relu"
```
```yaml
model:
  name: "QuantumGravSearchSpace"
  gcn_net:
    - in_dim: 12 # number of node features
      out_dim: [128, 256]
      dropout:
        type: tuple # to distinguish from categorical
        value: [0.2, 0.5, 0.1] # range for dropout, min, max, step
      gnn_layer_type: ["sage", "gcn", "gat", "gco"]
      normalizer: ["batch_norm", "identity", "layer_norm"]
      activation: ["relu", "leaky_relu", "sigmoid", "tanh", "identity"]
```

* Categorical parameters are defined by assigning the parameter name a list of possible values (`bool`, `string`, `float`, or `int`). In the example above, this applies to `out_dim`, `gnn_layer_type`, `normalizer`, and `activation`,
* Floating point and integer parameters are specified as a list of three items:
    * [`float`, `float`, `float` or `bool`] for floats
    * [`int`, `int`, `int`] for integers
    * To avoid confusion with categorical lists, the hyperparameter structure includes two sub-fields:
        * `type`: set to `"tuple"`
        * `value`: hold the 3-item tuple
    * Another example for floats:
        ```yaml
        learning_rate:
            type: tuple
            value: [1e-5, 1e-1, true]
        ```

Full example of the search space built based on the base setting from [the configuration `dict`](./training_a_model.md#the-configuration-dict):

```yaml
model:
  name: "QuantumGravSearchSpace"
  gcn_net:
    - in_dim: 12 # number of node features
      out_dim: [128, 256]
      dropout:
        type: tuple # to distinguish from categorical
        value: [0.2, 0.5, 0.1] # range for dropout, min, max, step
      gnn_layer_type: ["sage", "gcn", "gat", "gco"]
      normalizer: ["batch_norm", "identity", "layer_norm"]
      activation: ["relu", "leaky_relu", "sigmoid", "tanh", "identity"]
      norm_args: 
        - 128 # should match out_dim, manually set later
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
    - in_dim: 128 # should match previous layer's out_dim, manually set later
      out_dim: [256, 512]
      dropout:
        type: tuple
        value: [0.2, 0.5, 0.1]
      gnn_layer_type: ["sage", "gcn", "gat", "gco"]
      normalizer: ["batch_norm", "identity", "layer_norm"]
      activation: ["relu", "leaky_relu", "sigmoid", "tanh", "identity"]
      norm_args: 
        - 256 # should match out_dim, manually set later
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
    - in_dim: 256 # should match previous layer's out_dim, manually set later
      out_dim: [128, 256]
      dropout:
        type: tuple
        value: [0.2, 0.5, 0.1]
      gnn_layer_type: ["sage", "gcn", "gat", "gco"]
      normalizer: ["batch_norm", "identity", "layer_norm"]
      activation: ["relu", "leaky_relu", "sigmoid", "tanh", "identity"]
      norm_args: 
        - 128 # should match out_dim, manually set later
      gnn_layer_kwargs:
        normalize: False
        bias: True
        project: False
        root_weight: False
        aggr: "mean"
  pooling_layer: ["mean", "max", "sum"]
  classifier:
    input_dim: 128 # should match last gcn_net layer's out_dim, manually set later
    output_dims: 
      - 2 # number of classes in classification task
    hidden_dims: 
      - 48
      - 18
    activation: ["relu", "leaky_relu", "sigmoid", "tanh", "identity"]
    backbone_kwargs: [{}, {}]
    output_kwargs: [{}]
    activation_kwargs: [{ "inplace": False }]

training:
  seed: 42
  # training loop
  device: "cuda"
  early_stopping_patience: 5
  early_stopping_window: 7
  early_stopping_tol: 0.001
  early_stopping_metric: "f1_weighted"
  checkpoint_at: 2
  checkpoint_path: /path/to/where/the/intermediate/models/should/go
  # optimizer
  learning_rate:
    type: tuple
    value: [1e-5, 1e-1, true]
  weight_decay:
    type: tuple
    value: [1e-6, 1e-2, true]
  # training loader
  batch_size: [32, 64]
  num_workers: 12
  pin_memory: False
  drop_last: True
  num_epochs: [50, 100, 200]
  split: 0.8
validation: &valtest
  batch_size: 32
  num_workers: 12
  pin_memory: False
  drop_last: True
  shuffle: True
  persistent_workers: True
  split: 0.1
testing: *valtest
```

### Dependency mapping

Given the following hyperparameters in the search space (unrelated lines are substitude by `...` for simplicity):

```yaml
model:
  name: "QuantumGravSearchSpace"
  gcn_net:
    - in_dim: 12 # number of node features
      out_dim: [128, 256]
      ...
      norm_args: 
        - 128 # should match out_dim, manually set later
      gnn_layer_kwargs:
        ...
    - in_dim: 128 # should match previous layer's out_dim, manually set later
      out_dim: [256, 512]
      dropout:
        ...
```

In this example, the first argument of `norm_args` must match `out_dim`, and the `in_dim` of the second layer must match the `out_dim` of the first layer. YAML anchors (`&`) and aliases (`*`) would not help here as they reference static values, while hyperparameters are assigned dynamically by Optuna at runtime.

To handle such cases, we introduce another YAML file with the same structure as the search space file:

```yaml
model:
  gcn_net:
    # layer 0
    - norm_args: 
      - "model.gcn_net[0].out_dim"
    # layer 1
    - in_dim: "model.gcn_net[0].out_dim"
      norm_args: 
        ...
```

Here, the first value of `norm_args` in the first layer is set to `"model.gcn_net[0].out_dim"`, which points to the `out_dim` of the first element in `model` -> `gcn_net`. The same approach is used for the `in_dim` of the second layer.

To use this mapping, users must understand the search space file's structure and ensure the dependency mapping file follows it exactly.

Full example of dependency mapping file for the above search space file:

```yaml
model:
  gcn_net:
    # layer 0
    - norm_args: 
      - "model.gcn_net[0].out_dim"
    # layer 1
    - in_dim: "model.gcn_net[0].out_dim"
      norm_args: 
      - "model.gcn_net[1].out_dim"
    # layer 2
    - in_dim: "model.gcn_net[1].out_dim"
      norm_args: 
      - "model.gcn_net[2].out_dim"
  classifier:
    input_dim: "model.gcn_net[-1].out_dim"
```

## QGTune

## An example of tuning with QGTune