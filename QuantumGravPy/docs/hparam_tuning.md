# Hyperparameter Optimization with Optuna

The `QuantumGrav` Python package lets users customize hyperparameters when building, training, validating, and testing GNN models. Choosing the right values is crucial for model performance.

To accelerate this process, we developed `QGTune`, a subpackage that uses [Optuna](https://optuna.readthedocs.io/en/stable/index.html) to automatically find optimal hyperparameters for specific objectives (e.g. minimizing loss or maximizing accuracy).

## Define Optuna search space

To use Optuna, we first need to define the hyperparameter search space with methods from [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html), including:

* `suggest_categorical()`: suggest a value for the categorical parameter
* `suggest_float()`: suggest a value for the floating point parameter
* `suggest_int()`: suggest a value for the integer parameter

To define search space in `QuantumGrav`, users need three setting files:

* Base setting file: contains all configurations for using the `QuantumGrav` Python package (see [the configuration `dict`](./training_a_model.md#the-configuration-dict)). The hyperparameter values in this file will serve as defaults when users want to enable only a subset of the search space (see details in [Build Optuna search space](#build-optuna-search-space)).
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

Full example of the search space YAML file built based on the base setting from [the configuration `dict`](./training_a_model.md#the-configuration-dict):

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

## QGTune subpackage

Main purpose of the GQTune subpackage includes:

* Create an Optuna study from a config file (preferably YAML file)
* Build Optuna search space from the three described YAML files
* Save hyperparamter values of the best trial
* Save hyperparameter values of the best config

### Create an Optuna study

The input for creating an Optuna study is a configuration dictionary that should include essential keys like `"storage"`, `"study_name"`, and `"direction"`, for example:

```python
{
  "study_name": "quantum_grav_study", # name of the Optuna study
  "storage": "experiments/results.log", # only supports JournalStorage for multi-processing
  "direction": "minimize" # direction of optimization ("minimize" or "maximize")
}
```

If `storage` is assigned to `None` (or `null` in YAML file), the study will be saved with `optuna.storages.InMemoryStorage`, i.e. in RAM only until the Python session ends.

For simplicity while working with multi-processing, we only support storage with [Optuna's JournalStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html).

### Build an Optuna search space

To build an Optuna search space with `QGTune`, users can use `build_search_space_with_dependencies()` function as in the following example:

```python
from QGTune import tune

def objective(trial, tuning_config):
  search_space_file = tuning_config.get("search_space_path")
  depmap_file = tuning_config.get("dependency_mapping_path")
  tune_model = tuning_config.get("tune_model")
  tune_training = tuning_config.get("tune_training")
  base_config_file = tuning_config.get("base_settings_path")
  built_search_space_file = tuning_config.get("built_search_space_path")

  search_space = tune.build_search_space_with_dependencies(
        search_space_file,
        depmap_file,
        trial,
        tune_model=tune_model,
        tune_training=tune_training,
        base_settings_file=base_config_file,
        built_search_space_file=built_search_space_file,
    )
  
  ...
```

* `search_space` is a dictionary whose keys correspond to hyperparameter names.
* `objective` is the function that will be used later for optimization
* `trial` is an object of `optuna.trial.Trial`
* `tuning_config` serves as the configuration dictionary for `QGTune`, defined by users (see a full example at the end of section [Save best trial and best config](#save-best-trial-and-best-config))
* `search_space_file`, `depmap_file`, `base_config_file` are paths to the search space file, dependency file, and base config file, respectively. These paths can be specified in `tuning_config`.
* `tune_model`: whether to tune the hyperparameters associated with the `model` part of the search space
* `tune_training`: whether to tune the hyperparameters associated with the `training` part of the search space
* `built_search_space_file`: path to save the built search space. All hyperparameter values defined via `trial` suggestions will be recorded in this file as their initial suggestions. These values do not represent the best trial. This file serves as a reference for generating the best configuration later.

Note that the `base_config_file` is required if either `tune_model` or `tune_training` is `False`. In this case, hyperparameter values from the base settings will overwrite the corresponding part in the `search_space` dictionary.

### Save best trial and best config

After running all trials, users can save hyperparameter values of the best trial to a YAML file with `save_best_trial(study, out_file)` function.

However, hyperparameters in this saved file only cover for the ones with values defined via `trial` suggestions and not the ones with fixed values (e.g. `model.gcn_net[0].in_dim`).

Therefore, to save all values of utilized parameters, users can use `save_best_config()` function.

```python
def save_best_config(
    built_search_space_file: Path,
    best_trial_file: Path,
    depmap_file: Path,
    output_file: Path,
):
```

* `built_search_space_file` is the file created after running `build_search_space_with_dependencies()`
* `best_trial_file` created by `save_best_trial()`
* `depmap_file` is needed again to make sure that all parameter dependencies are resolved

Full example of tuning config YAML file used for `QGTune`

```yaml
tune_model: false # whether to use search space for model settings
tune_training: true # whether to use search space for training settings
base_settings_path: base_settings.yam # path to the base settings file
search_space_path: search_space.yaml # path to the search space config file
dependency_mapping_path: depmap.yaml # path to the dependency mapping file
built_search_space_path: built_search_space.yaml # path to save the built search space with dependencies applied
study_name: quantum_grav_study # name of the Optuna study
storage: experiments/results.log # storage file for the Optuna study, only supports JournalStorage for multi-processing
direction: minimize # direction of optimization ("minimize" or "maximize")
n_trials: 20 # number of trials for hyperparameter tuning
timeout: 600 # timeout in seconds for the study
n_jobs: 1 # number of parallel jobs for multi-threading (set to 1 for single-threaded)
n_processes: 4 # number of parallel processes for multi-processing, each process runs n_trials * n_iterations/n_processes
n_iterations: 8 # number of iterations to run the tuning process (each iteration runs n_trials)
best_trial_path: best_trial.yaml # path to save the best trial information
best_param_path: best_params.yaml # path to save the best hyperparameters
```

## An example of tuning with QGTune

We have provided an example in the [tune_example.py](./examples/tune_example.py) file to demonstrate the functionality of `QGTune`.

In this example, we created sample config for tuning, search space, dependency mapping, and base settings. A small model is also defined based on [Optuna's PyTorch example](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py).

The dataset used in this example is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). The task is to classify each 28×28 grayscale image into one of 10 classes.

To allow Optuna to track training progress, we need to call `trial.report` after each epoch:

```python
def objective(trial, tuning_config):
  ...
  search_space = ...
  ...
  # prepare model
  ...
  # prepare optimizer
  ...
  # prepare data
  ...
  for epoch in range(epochs):
    # train the model
    ...
    # validate the model
    ...
    accuracy = ...

    trial.report(accuracy, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
```

We also used Optuna's [multi-process optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-process-optimization) in this example.

## Notes on pruner and parallelization

We used [optuna.pruners.MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) when creating an Optuna study (`QGTune.tune.create_study()`). Support for additional pruners may be added in the future if required.

Although users can specify `n_jobs` (for multi-threading) when running a study optimization, we recommend keeping `n_jobs` set to `1`. According to [Optuna's Multi-thread Optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-thread-optimization):

> Multi-thread optimization has traditionally been inefficient in Python due to the Global Interpreter Lock (GIL). However, starting from Python 3.14 (pending official release), the GIL is expected to be removed. This change will make multi-threading a good option, especially for parallel optimization.