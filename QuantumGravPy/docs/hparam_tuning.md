# Hyperparameter Optimization

The `QuantumGrav` Python package lets users customize hyperparameters when building, training, validating, and testing GNN models. Choosing the right values is crucial for model performance.

To accelerate this process, we developed two ways to handle possible values of hyperparameters:

* Using custom YAML tags to generate a list of configs based on the cartesian product of possible values of hyperparameters, then run training on each config.
    * Pros: Fully deterministic, exhaustive, simple, transparent
    * Cons: Explodes combinatorially, no guided search
    * Useful if:
        * Our search space is small.
        * We need absolute transparency and determinism.
        * We are debugging or doing systematic experiments.
        * We want full coverage of combinations.
        * We avoid extra dependencies or complexity.
* Using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) to create hyperparameter search space on a config file then automatically find optimal hyperparameter values for specific objectives (e.g. minimizing loss or maximizing accuracy).
    * Pros: Support of intelligent search and pruning algorithms, visualization (when using with database), scalable running
    * Cons: Stochastic sampling, less exhaustive, more complex as we need to integrate the training loop into Optuna's objective function
    * Useful if:
        * The search space is moderately large or huge.
        * We want efficient, guided optimization.
        * We need early stopping for poor configs.
        * We run on clusters or need parallel trials.
        * We want good results with limited compute budget.

## Config handling with custom YAML tags

We developed the following custom YAML tags to specify possible values of hyperparamters:

* `!sweep` tag: Used to define possible categorical values for a hyperparameter. For example, if we want to experiment with two different values for the number of epochs (32 and 64), we can specify them as follows:
    ```yaml
    training:
        ...
        epochs: !sweep
            values: [32, 64]
    ```

* `!coupled-sweep` tag: If a hyperparameter’s values depend on another sweep hyperparameter, we can use coupled-sweep to link them. For example, if we want the batch size to be 64 when training for 32 epochs and 128 when training for 64 epochs, we can specify the batch size as follows:
    ```yaml
    training:
        ...
        batch_size: !coupled-sweep
            target: training.epochs
            values: [64, 128]
    ```

* `!range` tag: If a hyperparameter takes values within an int or float range, the range can be specified as below:
    ```yaml
    epochs: !range # int range
        start: 10
        stop: 30
        step: 5
    drop_rate: !range # float range
        start: 0.1
        stop: 0.5
        step: 0.2
    
    ```
    **Note: The stop value is included.**

* `!random_uniform` tag: This tag generates a set of float values uniformly sampled from a specified range. We can define the `size` (the number of values to produce); if not provided, it defaults to 5. When `log=True`, values are sampled in the log domain; otherwise, they are sampled linearly.
    ```yaml
    lr: !random_uniform # float values in log domain
        start: 1e-5
        stop: 1e-2
        log: true
        size: 7
    ```

* `!reference` tag: In some cases, a hyperparameter must share the same value as another. For instance, the input dimension of one layer and the output dimension of the previous layer. The `!reference` tag handles such cases:
    ```yaml
    model:
        layers:
            -
                in_dim: 728
                out_dim: 64
            -
                in_dim: !reference
                    target: model.layers[0].out_dim
    ```

* `!pyobject` tag: Ultimately, to specify a Python object in the config file, we use the `!pyobject` tag. This is useful for assigning a Python object to a model or layer type. For example:
    ```yaml
    model:
        conv_layer: !pyobject torch_geometric.nn.conv.sage_conv.SAGEConv
    ```

### ConfigHandler class

TODO: brief explanation.

## Optimization with Optuna

The second option for finding optimal hyperparameter values is to use Optuna together with the custom YAML tag.

We developed a subpackage `QGTune` to convert a config file with custom YAML tags to hyperparameter search space for Optuna. Main purpose of the GQTune subpackage includes:

* Build Optuna search space from a model/trainer config YAML file (with above custom tags)
* Create an Optuna study from a tuning config file (preferably YAML file)
* Save the best config with optimal hyperparameter values back to a YAML file.

### Define an Optuna search space using QGTune

#### Optuna suggestion with custom YAML tag

Optuna use [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) to define hyperparameter search space for each study. Main functions include:

* `suggest_categorical()`: suggest a value for a categorical parameter, which is a list of `bool`, `string`, `float`, or `int`
* `suggest_float()`: suggest a value within a range for a floating point parameter, in linear or log domain
* `suggest_int()`: suggest a value within a range for an integer parameter

The first suggestion function corresponds to the `!sweep` tag described above, while `suggest_int()` maps to the `!range` tag (when `start`, `stop`, and `step` are all integers). Float suggestions are handled through the `!range` and `!random_uniform` tags.

For example, the following configuration:

```yaml
epochs: !range
    start: 10
    stop: 30
    step: 5
```

is equivalent to calling `optuna.trial.Trial.suggest_int(param_name, start, stop, step)`. 

Here, `param_name` represents the parameter name stored in the Optuna study. We define this name as the path from the root node to the current node in the configuration file, e.g. `training.epochs` or `model.layer.0.in_dim`. Note that in this naming scheme, all parts are separated by dots (`.`). Unlike the `target` field in the `!reference` tag, no square brackets (`[]`) are used.

#### Create Optuna search space from a model/trainer config file

To build an Optuna search space with `QGTune`, we can use `build_search_space()` function as in the following example:

```python
from QuantumGrav.QGTune import tune

def objective(trial: optuna.trial.Trial, config_file: Path):

    search_space = tune.build_search_space(
        config_file=config_file,
        trial=trial,
    )
  ...
```

* `objective` is the function that will be used later by an Optuna study for optimization
* `trial` is an object of `optuna.trial.Trial`, representing trials of an Optuna study
* `config_file` is path to the model/trainer configuration file (see [the configuration `dict`](./training_a_model.md#the-configuration-dict) for an example)
* `search_space` is a dictionary constructed from the model/trainer config file, where values of hyperparameters defined with custom YAML tags are replaced by their corresponding Optuna trial suggestions. For example, the `epochs` configuration shown earlier would be transformed into:
    ```yaml
    training:
        epochs: optuna.trial.Trial.suggest_int("training.epochs", start, stop, step)
    ```

    Subsequently, this `search_space` dictionary would be used for creating a GNN model and training the model as described in [The Trainer class](./training_a_model.md#the-trainer-class).

### Create an Optuna study

The input for creating an Optuna study is a tuning configuration dictionary that should include essential keys like `"storage"`, `"study_name"`, and `"direction"`, for example:

```python
{
  "study_name": "quantum_grav_study", # name of the Optuna study
  "storage": "experiments/results.log", # only supports JournalStorage for multi-processing
  "direction": "minimize" # direction of optimization ("minimize" or "maximize")
}
```

If `storage` is assigned to `None` (or `null` in YAML file), the study will be saved with `optuna.storages.InMemoryStorage`, i.e. in RAM only until the Python session ends.

For simplicity while working with multi-processing, we only support storage with [Optuna's JournalStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html).

### Save the best config to a YAML file

After completing all trials, the hyperparameter values from the best trial, along with other fixed parameters, can be saved to a YAML file using the `save_best_config()` function.

```python
def save_best_config(
    config_file: Path,
    best_trial: optuna.trial.FrozenTrial,
    output_file: Path,
):
```

* `config_file` is the path to the model/trainer configuration file
* `best_trial` is the trial with the optimal results, obtained from an Optuna study.
* `output_file` is the path to the YAML output file

### An example of tuning with QGTune

We present here an example to demonstrate the functionality of `QGTune`.

#### Example explanation

In this example, we would:

* create sample configuration files for tuning and model/trainer settings
* define a small model based on [Optuna's PyTorch example](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py).
* load dat from [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. 
* define the Optuna `objective` function.
    * The task is to classify each 28×28 grayscale image into one of 10 fashion categories, such as T-shirt, coat, or sneaker.
    * To enable Optuna to monitor training progress, `trial.report()` should be called after each epoch to record the metric being optimized. In this example, we use `accuracy`.

      ```python
      def objective(trial: optuna.trial.Trial, config_file: Path):
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

* run the optimization process for a specified number of trials (`n_trials`)
* use Optuna's [multi-process optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-process-optimization) by setting the number of iterations (`n_iterations`) and processes (`n_processes`).

#### Example code

```python
# tune_example.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # make sure no GPU is used

from QuantumGrav.QGTune import tune
import optuna
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from optuna.storages.journal import JournalStorage, JournalFileBackend

DEVICE = torch.device("cpu")

current_dir = Path(__file__).parent
tmp_dir = current_dir / "tmp"


def create_tuning_config_file(tmp_path: Path) -> Path:
    """Create a tuning configuration YAML file for testing purposes."""
    # Note on n_jobs: Multi-thread optimization has traditionally been inefficient
    # in Python due to the Global Interpreter Lock (GIL) (Python < 3.14)
    tuning_config = {
        "config_file": str(tmp_path / "config.yaml"),
        "study_name": "test_study",
        "storage": str(tmp_path / "test_study.log"),
        "direction": "maximize",
        "n_trials": 15,
        "timeout": 600,
        "n_jobs": 1,  # set to >1 to enable multi-threading,
        "best_config_file": str(tmp_path / "best_config.yaml"),
        "n_processes": 2,
        "n_iterations": 2,
    }
    tuning_config_file = tmp_path / "tuning_config.yaml"
    with open(tuning_config_file, "w") as f:
        yaml.safe_dump(tuning_config, f)

    return tuning_config_file


def create_config_file(file_path: Path) -> Path:
    """Create a base configuration YAML file with tuning parameters for testing purposes."""
    yaml_text = """
        model:
            n_layers: 3
            nn:
                -
                    in_dim: 784
                    out_dim: !sweep
                        values: [128, 256]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
                -
                    in_dim: !reference
                        target: model.nn[0].out_dim
                    out_dim: !sweep
                        values: [16, 32]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
                -
                    in_dim: !reference
                        target: model.nn[1].out_dim
                    out_dim: !sweep
                        values: [16, 32]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
        training:
            batch_size: !sweep
                values: [16, 32]
            optimizer: !sweep
                values: ["Adam", "SGD"]
            lr: !random_uniform
                start: 1e-5
                stop: 1e-2
                log: true
            epochs: !sweep
                values: [2, 5, 7]
    """

    with open(file_path, "w") as f:
        f.write(yaml_text)


def define_small_model(config):
    n_layers = config["model"]["n_layers"]
    layers = []

    for i in range(n_layers):
        in_dim = config["model"]["nn"][i]["in_dim"]
        out_dim = config["model"]["nn"][i]["out_dim"]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        dropout = config["model"]["nn"][i]["dropout"]
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(out_dim, 10))  # classification of 10 classes
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def load_data(config, dir_path):
    batch_size = config["training"]["batch_size"]
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            dir_path, train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(dir_path, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial: optuna.trial.Trial, config_file: Path):

    search_space = tune.build_search_space(
        config_file=config_file,
        trial=trial,
    )

    # prepare model
    model = define_small_model(search_space).to(DEVICE)

    # prepare optimizer
    optimizer_name = search_space["training"]["optimizer"]
    lr = search_space["training"]["lr"]
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # prepare data
    data_dir = tmp_dir / "data"
    train_loader, valid_loader = load_data(search_space, data_dir)

    # train the model
    epochs = search_space["training"]["epochs"]
    batch_size = search_space["training"]["batch_size"]
    n_train_examples = batch_size * 30
    n_valid_examples = batch_size * 10

    # training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * batch_size > n_train_examples:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.NLLLoss()(output, target)
            loss.backward()
            optimizer.step()

        # validate the model
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                if batch_idx * batch_size > n_valid_examples:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        # report value to Optuna
        trial.report(accuracy, epoch)

        # prune trial if needed
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


def tune_integration(run_idx, tuning_config):  # run_idx is the iteration index

    study = tune.create_study(tuning_config)
    study.optimize(
        partial(objective, config_file=Path(tuning_config["config_file"])),
        n_trials=tuning_config["n_trials"],
        timeout=tuning_config["timeout"],
        n_jobs=tuning_config["n_jobs"],
    )

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    return (
        len(study.trials),
        len(pruned_trials),
        len(complete_trials),
        study.best_trial.value,
    )


if __name__ == "__main__":
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # create tuning config
    tuning_config = tune.load_yaml(create_tuning_config_file(tmp_dir))
    n_processes = tuning_config.get("n_processes", 1)
    n_iterations = tuning_config.get("n_iterations", 1)

    # create config file
    create_config_file(Path(tuning_config["config_file"]))

    print("Starting the tuning integration process...")
    with Pool(processes=n_processes) as pool:
        local_results = pool.map(
            partial(tune_integration, tuning_config=tuning_config), range(n_iterations)
        )

    print("Tuning results for each run:------------------")
    for i, result in enumerate(local_results):
        n_trials, n_pruned, n_complete, best_value = result
        print(f"Study statistics for run {i}: ")
        print("  Number of finished trials: ", n_trials)
        print("  Number of pruned trials: ", n_pruned)
        print("  Number of complete trials: ", n_complete)
        print("  Best trial value: ", best_value)

    storage = JournalStorage(
        JournalFileBackend(tuning_config["storage"])  # use uncompressed journal
    )
    study = optuna.load_study(study_name=tuning_config["study_name"], storage=storage)

    print("Best trial global:----------------------------")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("Saving the best configuration...")
    tune.save_best_config(
        config_file=Path(tuning_config["config_file"]),
        best_trial=trial,
        output_file=Path(tuning_config["best_config_file"]),
    )
```

#### Example result

```bash
Starting the tuning integration process...
[I 2025-11-07 11:24:36,912] A new study created in Journal with name: test_study
Study test_study was created and saved to /QuantumGrav/QuantumGravPy/docs/tmp/test_study.log.
[I 2025-11-07 11:24:36,946] Using an existing study with name 'test_study' instead of creating a new one.
Study test_study was created and saved to /QuantumGrav/QuantumGravPy/docs/tmp/test_study.log.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.4M/26.4M [00:00<00:00, 57.6MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26.4M/26.4M [00:00<00:00, 54.9MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29.5k/29.5k [00:00<00:00, 4.39MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29.5k/29.5k [00:00<00:00, 4.53MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.42M/4.42M [00:00<00:00, 43.7MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.42M/4.42M [00:00<00:00, 42.4MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.15k/5.15k [00:00<00:00, 88.9MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.15k/5.15k [00:00<00:00, 40.9MB/s]
[I 2025-11-07 11:24:38,433] Trial 1 finished with value: 0.625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0016765190716563534, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:38,632] Trial 2 finished with value: 0.075 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.30000000000000004, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'SGD', 'training.lr': 0.0002146791042032724, 'training.epochs': 2}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:38,666] Trial 0 finished with value: 0.08125 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.2, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.30000000000000004, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.5, 'training.batch_size': 32, 'training.optimizer': 'SGD', 'training.lr': 0.0003945248169221987, 'training.epochs': 7}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:38,894] Trial 3 finished with value: 0.0875 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.2, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.4, 'training.batch_size': 16, 'training.optimizer': 'SGD', 'training.lr': 0.00018633239421692437, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:38,898] Trial 4 finished with value: 0.090625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.2, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.4, 'training.batch_size': 32, 'training.optimizer': 'SGD', 'training.lr': 0.00379972665577471, 'training.epochs': 2}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,314] Trial 6 finished with value: 0.13125 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.2, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'SGD', 'training.lr': 0.0008484799816474145, 'training.epochs': 7}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,435] Trial 5 finished with value: 0.1375 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.30000000000000004, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.30000000000000004, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'SGD', 'training.lr': 8.354945492960286e-05, 'training.epochs': 7}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,641] Trial 8 finished with value: 0.09375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.30000000000000004, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.5, 'training.batch_size': 32, 'training.optimizer': 'SGD', 'training.lr': 8.08272287524503e-05, 'training.epochs': 2}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,718] Trial 7 finished with value: 0.11875 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.30000000000000004, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.2, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.5, 'training.batch_size': 16, 'training.optimizer': 'SGD', 'training.lr': 0.0006188877342998116, 'training.epochs': 7}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,822] Trial 9 finished with value: 0.103125 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.5, 'training.batch_size': 32, 'training.optimizer': 'SGD', 'training.lr': 0.0008168448454213926, 'training.epochs': 2}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:39,913] Trial 10 finished with value: 0.134375 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.30000000000000004, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.2, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.5, 'training.batch_size': 32, 'training.optimizer': 'SGD', 'training.lr': 0.00021660920841439104, 'training.epochs': 2}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:40,157] Trial 11 finished with value: 0.18125 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.00883615551351022, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:40,269] Trial 12 finished with value: 0.15625 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.30000000000000004, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 1.5473312399729955e-05, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:40,531] Trial 13 finished with value: 0.29375 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.009542442262415065, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:40,610] Trial 14 finished with value: 0.40625 and parameters: {'model.nn.0.out_dim': 256, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.008577489282530437, 'training.epochs': 5}. Best is trial 1 with value: 0.625.
[I 2025-11-07 11:24:40,855] Trial 15 finished with value: 0.6375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.002801697167722077, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:40,924] Trial 16 finished with value: 0.5375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0024869223253056747, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,217] Trial 17 finished with value: 0.5 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0017531394038905789, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,311] Trial 18 finished with value: 0.55625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0020300569180315316, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,545] Trial 19 finished with value: 0.54375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.4, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0015706471695032163, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,628] Trial 20 finished with value: 0.5875 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.4, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0035034963024678494, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,852] Trial 21 finished with value: 0.60625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.004692563836368309, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:41,949] Trial 22 finished with value: 0.475 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.5, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.4, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.00413377148937998, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:42,158] Trial 23 finished with value: 0.5375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.004366383387329541, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:42,268] Trial 24 finished with value: 0.55625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.005176248318654213, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:42,493] Trial 25 finished with value: 0.51875 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0011242281028770243, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:42,577] Trial 26 finished with value: 0.50625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.4, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.001187102582494059, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:43,038] Trial 27 finished with value: 0.44375 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.30000000000000004, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 32, 'training.optimizer': 'Adam', 'training.lr': 0.0013102961720892286, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:43,132] Trial 28 finished with value: 0.465625 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.4, 'model.nn.1.out_dim': 16, 'model.nn.1.dropout': 0.30000000000000004, 'model.nn.2.out_dim': 16, 'model.nn.2.dropout': 0.30000000000000004, 'training.batch_size': 32, 'training.optimizer': 'Adam', 'training.lr': 0.0005325671025809822, 'training.epochs': 5}. Best is trial 15 with value: 0.6375.
[I 2025-11-07 11:24:43,504] Trial 29 finished with value: 0.7125 and parameters: {'model.nn.0.out_dim': 128, 'model.nn.0.dropout': 0.2, 'model.nn.1.out_dim': 32, 'model.nn.1.dropout': 0.5, 'model.nn.2.out_dim': 32, 'model.nn.2.dropout': 0.2, 'training.batch_size': 16, 'training.optimizer': 'Adam', 'training.lr': 0.0026496075439612654, 'training.epochs': 7}. Best is trial 29 with value: 0.7125.
Tuning results for each run:------------------
Study statistics for run 0: 
  Number of finished trials:  29
  Number of pruned trials:  0
  Number of complete trials:  28
  Best trial value:  0.6375
Study statistics for run 1: 
  Number of finished trials:  30
  Number of pruned trials:  0
  Number of complete trials:  30
  Best trial value:  0.7125
Best trial global:----------------------------
  Value:  0.7125
  Params: 
    model.nn.0.out_dim: 128
    model.nn.0.dropout: 0.2
    model.nn.1.out_dim: 32
    model.nn.1.dropout: 0.5
    model.nn.2.out_dim: 32
    model.nn.2.dropout: 0.2
    training.batch_size: 16
    training.optimizer: Adam
    training.lr: 0.0026496075439612654
    training.epochs: 7
Saving the best configuration...
Best configuration saved to /QuantumGrav/QuantumGravPy/docs/tmp/best_config.yaml.
```

#### Notes on pruner and parallelization

We used [optuna.pruners.MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) when creating an Optuna study (`QGTune.tune.create_study()`). Support for additional pruners may be added in the future if required.

Although users can specify `n_jobs` (for multi-threading) when running a study optimization, we recommend keeping `n_jobs` set to `1`, according to [Optuna's Multi-thread Optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#multi-thread-optimization).