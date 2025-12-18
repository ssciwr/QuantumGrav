# Training a Model

After the data processing, we can set up the training process. This is done using the `Trainer` class.

The `Trainer` class follows a pattern in which code and training parameters are separated: It expects a dictionary containing all the parameters, and a set of objects that take care of evaluation of the training process.

The config `dict` allows us to store the parameters in an external file (YAML would be the preferable option) and read it in from there, such that we can have different configs for different runs that can be stored alongside the experiments. This is helpful for reproducibility of experiments.

## The Trainer class
`Trainer` is fully config-driven. Build a config dict (or YAML) matching `Trainer.schema`, then construct with `Trainer.from_config(config)`. The loss (`criterion`), optional `apply_model`, validation/test evaluators, early stopping, optimizer, learning rate schedulers and data loading are all specified in the config.

The `criterion` must be callable as `criterion(outputs, data)` and return a scalar tensor. If your model has a custom forward signature, set `apply_model` in the config.

## The evaluators
The next three arguments are needed to evaluate and test the model and to implement a stopping criterion, so they deserve their own little section.

### `Validator` and `Tester`
`Validator` and `Tester` inherit from `Evaluator`. Both compute metrics via `evaluate(model, data_loader)` and log via `report(data)`. Configure them in the Trainer config under `validation.validator` and `testing.tester`.

You can attach custom monitors with `evaluator_tasks` (each task has a `name` and a `monitor` callable). Monitors receive collected predictions/targets and compute metrics such as F1.

### Early stopping
Configure `DefaultEarlyStopping` via `early_stopping` in the Trainer config. Define per-task settings (`metric`, `delta`, `mode`, `grace_period`, `init_best_score`) and a global `patience` and aggregation `mode` (`any`/`all`). It consumes the validator’s dataframe to decide when to stop.

## The configuration `dict`
This provides all the necessary parameters for a training run, and as such has a fixed structure. This is best shown with an example:
```yaml
name: "QuantumGravRun"
training:
    seed: 42
    device: "cuda"
    path: /path/to/run_artifacts
    num_epochs: 50
    batch_size: 32
    optimizer_type: !pyobject torch.optim.Adam
    optimizer_args: []
    optimizer_kwargs:
        lr: 0.001
        weight_decay: 0.0001
    num_workers: 4
    pin_memory: true
    drop_last: false
    checkpoint_at: 5
    shuffle: true
data:
    files: ["/path/to/data1.zarr", "/path/to/data2.zarr"]
    output: "/path/to/output"
    reader: !pyobject mymodule.reader
    pre_transform: !pyobject mymodule.pre_transform
    pre_filter: !pyobject mymodule.pre_filter
    validate_data: true
    n_processes: 4
    chunksize: 1000
    split: [0.8, 0.1, 0.1]
    shuffle: false
model:
    encoder_type: !pyobject QuantumGrav.models.Sequential
    encoder_kwargs:
        layers:
            - [!pyobject torch_geometric.nn.SAGEConv, [12, 64], {"aggr": "mean"}, "x, edge_index"]
            - [!pyobject torch.nn.ReLU, [], {}, "x"]
            - [!pyobject torch_geometric.nn.SAGEConv, [64, 64], {"aggr": "mean"}, "x, edge_index"]
        forward_signature: "x, edge_index, batch"
        with_skip: false
    pooling_layers:
        - [!pyobject torch_geometric.nn.global_mean_pool, [], {}]
    aggregate_pooling_type: !pyobject torch.cat
    graph_features_net_type: !pyobject QuantumGrav.models.LinearSequential
    graph_features_net_kwargs:
        dims: [[4, 32]]
        activations: [!pyobject torch.nn.ReLU]
    aggregate_graph_features_type: !pyobject torch.cat
    downstream_tasks:
        - [!pyobject QuantumGrav.models.LinearSequential, null, {"dims": [[96, 128], [128, 3]], "activations": [!pyobject torch.nn.ReLU, !pyobject torch.nn.Identity]}]
validation:
    batch_size: 32
    validator:
        device: "cuda"
        criterion: !pyobject mymodule.compute_loss
        evaluator_tasks:
            - name: "f1_weighted"
                monitor: !pyobject sklearn.metrics.f1_score
                kwargs:
                    average: "weighted"
testing:
    batch_size: 32
    tester:
        device: "cuda"
        criterion: !pyobject mymodule.compute_loss
        evaluator_tasks:
            - name: "loss_avg"
                monitor: !pyobject numpy.mean
early_stopping:
    tasks:
        0:
            metric: "loss_avg"
            delta: 1e-4
            grace_period: 2
            init_best_score: 1e9
            mode: "min"
    mode: "any"
    patience: 5
criterion: !pyobject mymodule.compute_loss
```
```
The config includes `name`, `training`, `data`, `model`, `validation`, `testing`, `early_stopping`, and `criterion`. See [`Graph Neural Network models`](./models.md) for model schema details.

## Train a model

The following is a complete end-to-end example for model training for a classification task. We are putting toghether the content from ['Using Datasets for data processing and batching'](./datasets_and_preprocessing.md) and from ['Training a model'](./training_a_model.md) and are overwriting the Evaluators to report F1 scores. Then, we set up the trainer class, prepare everything and run training. For completeness, we put everything into on file here, but it may be advisable to split your script into multiple files if you write as much code as here. Also, we might add several variants of evaluators as default in the future. To get a good idea of how the system works, please work through this example carefully and make sure you understand each step.
For bugs or issues, please report them [here](https://github.com/ssciwr/QuantumGrav/issues).

### Minimal example

```python
import QuantumGrav as QG
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import zarr, numpy as np

def reader(store, idx, float_dtype, int_dtype, validate):
    g = zarr.open_group(store.root)
    cset_size = int(g["cset_size"][idx])
    adj = np.transpose(g["adjacency_matrix"][idx, :, :])[:cset_size, :cset_size]
    edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))
    x = torch.tensor(g["max_pathlens_future"][idx, :][:cset_size], dtype=float_dtype).unsqueeze(1)
    y = torch.tensor(int(g["manifold_like"][idx]), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

def compute_loss(outputs, data):
    return torch.nn.CrossEntropyLoss()(outputs[0], data.y)

cfg = {  # use YAML with !pyobject tags for types if preferred
  "name": "run",
  "training": {"seed": 0, "device": "cuda", "path": "/tmp/qg", "num_epochs": 5, "batch_size": 32,
                "optimizer_type": torch.optim.Adam, "optimizer_args": [], "optimizer_kwargs": {"lr": 1e-3},
                "num_workers": 0, "pin_memory": True, "drop_last": False, "checkpoint_at": 2, "shuffle": True},
  "data": {"files": ["/path/to/data.zarr"], "output": "/tmp/qg/output", "reader": reader,
            "validate_data": True, "n_processes": 1, "chunksize": 1000, "split": [0.8, 0.1, 0.1]},
  "model": {"encoder_type": QG.models.Sequential, "encoder_kwargs": {"layers": [[torch_geometric.nn.SAGEConv, [12, 64], {"aggr": "mean"}, "x, edge_index"], [torch.nn.ReLU, [], {}, "x"]], "forward_signature": "x, edge_index, batch"},
             "pooling_layers": [[torch_geometric.nn.global_mean_pool, [], {}]], "aggregate_pooling_type": torch.cat,
             "downstream_tasks": [[QG.models.LinearSequential, None, {"dims": [[64, 3]], "activations": [torch.nn.Identity]}]]},
  "validation": {"batch_size": 32, "validator": {"device": "cuda", "criterion": compute_loss, "evaluator_tasks": []}},
  "testing": {"batch_size": 32, "tester": {"device": "cuda", "criterion": compute_loss, "evaluator_tasks": []}},
  "early_stopping": {"tasks": {0: {"metric": "loss_avg", "delta": 1e-4, "grace_period": 2, "init_best_score": 1e9, "mode": "min"}}, "mode": "any", "patience": 5},
  "criterion": compute_loss,
}

trainer = QG.Trainer.from_config(cfg)
train_loader, val_loader, test_loader = trainer.prepare_dataloaders()
training_result, validation_result = trainer.run_training(train_loader, val_loader)
test_result = trainer.run_test(test_loader)
```