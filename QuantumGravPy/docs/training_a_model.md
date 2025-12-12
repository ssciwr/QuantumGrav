# Training a Model

After the data processing, we can set up the training process. This is done using the `Trainer` class.

The `Trainer` class follows a pattern in which code and training parameters are separated: It expects a dictionary containing all the parameters, and a set of objects that take care of evaluation of the training process.

The config `dict` allows us to store the parameters in an external file (YAML would be the preferable option) and read it in from there, such that we can have different configs for different runs that can be stored alongside the experiments. This is helpful for reproducibility of experiments.

## The Trainer class
Let's have a look at how the trainer class works first. It's constructor reads:

```python
class Trainer:
    """Trainer class for training and evaluating GNN models."""

    def __init__(
        self,
        config: dict[str, Any],
        # training and evaluation functions
        criterion: Callable,
        apply_model: Callable | None = None,
        # training evaluation and reporting
        early_stopping: Callable[[Collection[Any] | torch.Tensor], bool]
        | None = None,
        validator: Validator | None = None,
        tester: Tester | None = None,
    )
```
The `config` argument has its own section [below](#the-configuration-dict). The `criterion` is a loss function that must have the following signature:
```python
criterion[[model_output, torch_geometric.data.Data], torch.Tensor]
```
i.e., it allows for passing in arbitrary model outputs and the original data object (which contains the target for supervised learning for instance) and returns a `torch.Tensor` which contains a scalar. The second argument `data` can be ignored if it's not needed.

The `apply_model` function is optional and defines how to call the model with the data. If you choose to use this class with your own model class, this might come in handy, especially if you change the signature of the model's `forward` method.

## The evaluators
The next three arguments are needed to evaluate and test the model and to implement a stopping criterion, so they deserve their own little section.

### `tester` and `validator`
We will start from the end, beginning with `tester`. This is an object of type `Tester` (an instance thereof or an object derived from it).
This class is build like this:

```python
class Tester(Evaluator):
    def __init__(
        self, device, criterion: Callable, apply_model: Callable | None = None
    )
```

i.e., it takes a loss function and an optional function to apply the model to data as well as the device to run on as constructor input. It then has a `test` function:

```python

    def test(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,
    ):
```

which applies the model to the data in the passed `DataLoader` object using the `criterion` function. This will then be passed to a `report` function:

```python
def report(self, data: list | pd.Series | torch.Tensor | np.ndarray)
```

This is a function or callable object that decides when to stop the training process based on a metric it gets. It eats a list or other iterable that contains the output fo applying the model to the data using `criterion`, and, by default, computes the mean and standard deviation thereof and reports them to standard out.

The `validator` object works exactly the same, (`Tester` and `Validator` have the same base class: `Evaluator`), so it's not documented separately here. The only difference is that what's called `test` in `Tester` is called `validate` in `Evaluator`.

Under the hood, they both use the `evalute` function of their parent class `Evaluator`. So in your derived classes, you can also overwrite that and then build your own type hierarcy on top of it.

The idea behind these two classes is that the user derives their own class from them and adjusts the `test`, `validate` and `report` functions to their own needs, e.g., for reporting the F1 score in a classification task.
Note that you can also break the type annotation if you think you need and for instance use your own evaluator classes - just make sure the call signatures are correct.

### `early_stopping`
This function is there to stop training once a certain condition has been reached. It eats the output of `Validator` and then computes a boolean that tells the `Trainer` object whether it should stop or continue training.

For example, this can look like the following code block, where we use a moving average over the loss and a tolerance to determine if we should stop or not: Each epoch, the mean validation loss over the window is evaluated and compared to the current best one, and if it's above that we subtract 1 from the `patience` variable. if `patience` runs out, we stop training. When we find a better best mean loss, we reset patience and continue.

Note that we are not using a function here, but a callable object. This is useful for cases where your `early stopping` logic has parameters that need to be held somewhere.

```python
class EarlyStopping(DefaultEarlyStopping):

    def __init__(
        self, patience: int, delta: float = 1e-4, window=7, metric: str = "loss"
    ):
        super().__init__(patience, delta, window)

        self.metric = metric
        self.logger = logging.getLogger("QuantumGrav.EarlyStopping")

    def __call__(self, data: pd.DataFrame | list[dict[Any]]) -> bool: # make it a callable object
        if isinstance(data, pd.DataFrame) is False:
            data = pd.DataFrame(data)

        window = min(self.window, len(data))
        smoothed = data[self.metric].rolling(window=window, min_periods=1).mean()
        if smoothed.iloc[-1] < self.best_score - self.delta:
            self.best_score = smoothed.iloc[-1]
            self.current_patience = self.patience
        else:
            self.current_patience -= 1
        self.logger.info(
            f"EarlyStopping: current patience: {self.current_patience}, best score: {self.best_score}, smoothed metric: {smoothed.iloc[-1]}"
        )

        return self.current_patience <= 0
```
This follows a similar principle to the other Evaluators: Derive from a common baseclass and overwrite the relevant methods with your own logic. In writing this, you need to make sure that the input of the `__call__` method must be compatible with the ouptut of the `validate` method of the `Validator` class.

## The configuration `dict`
This provides all the necessary parameters for a training run, and as such has a fixed structure. This is best shown with an example:
```yaml
model:
  name: "QuantumGravTest"
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
  learning_rate: 0.001
  weight_decay: 0.0001
  # training loader
  batch_size: 32
  num_workers: 12
  pin_memory: False
  drop_last: True
  num_epochs: 200
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
A config **must** have the high-level nodes 'model', 'training', 'validation', and 'testing'. Data processing is currently not governed by a config because it's too specialized for the task at hand.
All the nodes in `training` above are necessary and cannot be left out, because they are needed for the DataLoaders and the evaluation and training semantics.

The `model` part defines the architecture of the model that's used, so please check the [`Graph Neural Network models`](./models.md) section again for how that works.

## Train a model

The following is a complete end-to-end example for model training for a classification task. We are putting toghether the content from ['Using Datasets for data processing and batching'](./datasets_and_preprocessing.md) and from ['Training a model'](./training_a_model.md) and are overwriting the Evaluators to report F1 scores. Then, we set up the trainer class, prepare everything and run training. For completeness, we put everything into on file here, but it may be advisable to split your script into multiple files if you write as much code as here. Also, we might add several variants of evaluators as default in the future. To get a good idea of how the system works, please work through this example carefully and make sure you understand each step.
For bugs or issues, please report them [here](https://github.com/ssciwr/QuantumGrav/issues).

### Full code example

```python
from pathlib import Path
import QuantumGrav as QG

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import zarr
import numpy as np
import shutil
import yaml
from sklearn.metrics import f1_score
import pandas as pd
import pickle
import logging

# set up logger
logging.basicConfig(
    level=logging.INFO,  # or logging.INFO if you want less verbosity
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


################################################################################
# data processing
# Find all .h5 files in the given directory and its subdirectories
def find_files(directory: Path, file_list: list[Path] = []):
    for path in directory.iterdir():
        if path.is_dir():
            find_files(path, file_list)
        else:
            if path.suffix == ".h5" and "backup" not in path.name:
                file_list.append(path)


# load data from zarr file and put it into a dictionary for further processing
# This is a custom reader function for the QGDataset
def load_data(
    file: zarr.Group,
    idx: int,
    float_dtype: torch.dtype,
    int_dtype: torch.dtype,
    validate_data: bool,
) -> dict:

    # get the hdf5 group to load data from. We assume there's a group 'adjacency_matrix' because all data is derived from the adjacency matrix of the causal set
    group = file["adjacency_matrix"]

    data = dict()

    # Load edge index and edge weight
    cset_size = torch.tensor(group["cset_size"][idx]).to(int_dtype)
    adj_raw = group["adjacency_matrix"][idx, :, :]

    data["adjacency_matrix"] = adj_raw
    data["cset_size"] = cset_size

    # target
    manifold_like = group["manifold_like"][idx]

    # features
    data["manifold_like"] = manifold_like
    feature_names = [
        "max_pathlens_future",
        "max_pathlens_past",
        "in_degree",
        "out_degree",
    ]
    data["feature_names"] = feature_names

    for feature_name in feature_names:
        # all features are vectors of size cset_size
        data[feature_name] = group[feature_name][idx, :][0:cset_size]

        # make features into correct shape and replace NaNs
        values = torch.tensor(
            group[feature_name][idx, :][0:cset_size], dtype=float_dtype
        ).unsqueeze(1)
        nan_found = torch.isnan(values)

        # replace NaNs with 0.0
        if nan_found.any():
            values = torch.nan_to_num(values, nan=0.0)
        data[feature_name] = values

    # return dictionary of raw data instead of full data object --> better separation of concerns here
    return data

# function for turning the raw data into a Data object that will be saved on disk
def pre_transform(data: dict) -> Data:
    """Pre-transform the data dictionary into a PyG Data object."""
    adjacency_matrix = data["adjacency_matrix"]
    cset_size = data["cset_size"]

    # this is a workaround for the fact that the adjacency matrix is stored in a transposed form when going from julia to hdf5
    adjacency_matrix = np.transpose(adjacency_matrix)
    adjacency_matrix = adjacency_matrix[0:cset_size, 0:cset_size]
    edge_index, edge_weight = dense_to_sparse(
        torch.tensor(adjacency_matrix, dtype=torch.float32)
    )

    # make node features
    node_features = []
    for feature_name in data["feature_names"]:
        node_features.append(data[feature_name])
    x = torch.cat(node_features, dim=1).to(torch.float32)

    # make targets
    y = torch.tensor(data["manifold_like"]).to(torch.long)

    # make data object
    tgdata = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y,
    )

    if not tgdata.validate():
        raise ValueError(f"Data validation failed for index {idx}.")
    return tgdata


################################################################################
# Testing and Validation helper classes.
class Validator(QG.Validator):
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
        prefix: str = "",
    ):
        super().__init__(device, criterion, apply_model)
        self.prefix = prefix
        self.data = pd.DataFrame(
            columns=[
                "avg_loss",
                "std_loss",
                "f1_per_class",
                "f1_unweighted",
                "f1_weighted",
            ],
        )

    # overwrite the default 'evaluate' function
    def evaluate(self, model, data_loader):
        model.eval()
        current_data = pd.DataFrame(
            np.nan,
            columns=["loss", "output", "target"],
            index=pd.RangeIndex(
                start=0, stop=len(data_loader) * data_loader.batch_size, step=1
            ),
        )
        start = 0
        stop = 0

        # evaluate the model and produce data we need
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data = batch.to(self.device)
                if self.apply_model:
                    outputs = self.apply_model(model, data)
                else:
                    outputs = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(outputs, data)
                manifold_like = outputs[0].argmax(dim=1).cpu().numpy()

                if loss.isnan().any():
                    print(f"NaN loss encountered in batch {i}.")
                    continue

                if np.isnan(manifold_like).any():
                    print(f"NaN manifold_like encountered in batch {i}.")
                    continue

                if torch.isnan(data.y).any():
                    print(f"NaN target encountered in batch {i}.")
                    continue
                stop = start + data.num_graphs

                current_data.iloc[start:stop, current_data.columns.get_loc("loss")] = (
                    loss.item()
                )

                current_data.iloc[
                    start:stop, current_data.columns.get_loc("output")
                ] = manifold_like

                current_data.iloc[
                    start:stop, current_data.columns.get_loc("target")
                ] = data.y.cpu().numpy()
                start = stop

        return current_data

    def report(self, data):
        # compute avg loss and F1 score and report via print
        per_class = f1_score(data["output"], data["target"], average=None)
        unweighted = f1_score(data["output"], data["target"], average="macro")
        weighted = f1_score(data["output"], data["target"], average="weighted")
        avg_loss = data["loss"].mean()
        std_loss = data["loss"].std()
        self.logger.info(f"{self.prefix} avg loss: {avg_loss:.4f} +/- {std_loss:.4f}")
        self.logger.info(f"{self.prefix} f1 score per class: {per_class}")
        self.logger.info(f"{self.prefix} f1 score unweighted: {unweighted}")
        self.logger.info(f"{self.prefix} f1 score weighted: {weighted}")

        self.data.loc[len(self.data)] = [
            avg_loss,
            std_loss,
            per_class,
            unweighted,
            weighted,
        ]

    def validate(self, model, data_loader):
        return self.evaluate(model, data_loader)

class Tester(Validator): # this still works, even though it breaks the type annotation for `tester`.
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        super().__init__(device, criterion, apply_model, prefix="Testing")

    def test(self, model, data_loader):
        return super().evaluate(model, data_loader)


################################################################################
# functions for the training loop
# loss function --> 'criterion'
def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    loss = torch.nn.CrossEntropyLoss()(
        x[0], data.y
    )  # one task -> use x[0] for the output

    if loss.isnan().any():
        raise ValueError(f"Loss contains NaN values. {x[0]} {data.y}")

    return loss

################################################################################
# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore over a set window. You can set this up for looking at the F1 score too for instance by changing the value for 'metric'. In this case, we are using a dataframe to collect the validation output.
class EarlyStopping(DefaultEarlyStopping):

    def __init__(
        self, patience: int, delta: float = 1e-4, window=7, metric: str = "loss"
    ):
        super().__init__(patience, delta, window)

        self.metric = metric
        self.logger = logging.getLogger("QuantumGrav.EarlyStopping")

    # we are using a dataframe. this works because the Validator class creates one.
    def __call__(self, data: pd.DataFrame | list[dict[Any, Any]]) -> bool: # make it a callable object
        window = min(self.window, len(data))
        smoothed = data[self.metric].rolling(window=window, min_periods=1).mean()

        if isinstance(data, pd.DataFrame) is False:
            data = pd.DataFrame(data)

        if smoothed.iloc[-1] < self.best_score - self.delta:
            self.best_score = smoothed.iloc[-1]
            self.current_patience = self.patience
        else:
            self.current_patience -= 1
        self.logger.info(
            f"EarlyStopping: current patience: {self.current_patience}, best score: {self.best_score}, smoothed metric: {smoothed.iloc[-1]}"
        )

        return self.current_patience <= 0

# apply model function if necessary. Here, this is not needed, because we have no edge features as such


################################################################################
# main function putting everything together
def main(path_to_data: str | Path | None, path_to_config: str | Path | None):
    logger = logging.getLogger("QuantumGrav")

    if path_to_data is None:
        raise ValueError("Path to data must be provided.")

    if path_to_config is not None:
        run_training = True
        with open(path_to_config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = None
        run_training = False
        logger.info("No config provided, only processing data")

    h5files = []
    find_files(Path(path_to_data), h5files)
    logger.info(f"Found {len(h5files)} files.")

    # augment files if necessary with 'num_causal_sets' dataset
    for file in h5files:
        with h5py.File(file, "r+") as f:
            if "num_causal_sets" not in f:
                logger.info("Adding num_causal_sets to file.")
                if (Path(file).parent / "backup.h5").exists() is False:
                    shutil.copy(file, Path(file).parent / "backup.h5")
                f["num_causal_sets"] = f["adjacency_matrix"]["adjacency_matrix"].shape[
                    0
                ]

    # create a dataset
    dataset = QG.QGDataset(
        input=h5files,
        output=path_to_data,
        reader=load_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=6,
        chunksize=200,
        transform=None,
        pre_transform=pre_transform,
        pre_filter=None,
    )


    if run_training and config is not None:
        logger.info(f"Running training with config: {path_to_config}")

        # instantiate all the evaluators
        validator = Validator(
            device=torch.device(config["training"]["device"]),
            criterion=compute_loss,
            apply_model=None,  # No need for apply_model in this case
        )
        tester = Tester(
            device=torch.device(config["training"]["device"]),
            criterion=compute_loss,
            apply_model=None,  # No need for apply_model in this case
        )

        early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping_patience"],
            delta=config["training"].get("early_stopping_tol", 1e-4),
            window=config["training"].get("early_stopping_window", 7),
            metric=config["training"].get("early_stopping_metric", "f1_weighted"),
        )

        # set up trainer class
        logger.info(f"Using device: {config['training']['device']}")
        logger.info("Building trainer")
        trainer = QG.Trainer(
            config,
            compute_loss,
            validator=validator,
            tester=tester,
            early_stopping=early_stopping,
            apply_model=None,  # No need for apply_model in this case
        )

        # prepare data loaders
        train_split = config["training"].get("split", 0.8)
        test_split = config["testing"].get("split", 0.1)
        val_split = config["validation"].get("split", 0.1)

        train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
            dataset, split=[train_split, val_split, test_split]
        )

        # initialize the model and optimizer
        trainer.initialize_model()

        trainer.initialize_optimizer()

        # run training
        training_result, validation_result = trainer.run_training(
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # test the model and return the result
        test_result = trainer.run_test(test_loader=test_loader)
        return training_result, validation_result, test_result
    else:
        return dataset


if __name__ == "__main__":

    path_to_data = Path(
        "path/to/hdf5_datafiles
    )
    # load config from file
    path_to_config = Path(
        "path/to/training_config.yaml"
    )
    # execute training and save result data
    train_data, valid_data, test_data = main(path_to_data, path_to_config)

    # save result data
    with open(path_to_data / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(path_to_data / "valid_data.pkl", "wb") as f:
        pickle.dump(valid_data, f)

    with open(path_to_data / "test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
```