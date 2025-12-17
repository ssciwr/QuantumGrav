from collections.abc import Collection
from typing import Any, Tuple, Dict

import numpy as np
from pathlib import Path
import logging
import tqdm
import yaml
from datetime import datetime
from math import ceil, floor
import jsonschema
import pandas as pd

from . import evaluate
from . import early_stopping
from . import gnn_model
from . import dataset_ondisk
from . import base

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import optuna


class Trainer(base.Configurable):
    """Trainer class for training and evaluating GNN models."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Model trainer class Configuration",
        "type": "object",
        "definitions": {
            "constructor": {
                "type": "object",
                "description": "Python constructor spec: type(*args, **kwargs)",
                "properties": {
                    "type": {
                        "description": "Fully-qualified import path or name of the type/callable to initialize",
                    },
                    "args": {
                        "type": "array",
                        "description": "Positional arguments for constructor",
                        "items": {},
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Keyword arguments for constructor",
                        "additionalProperties": {},
                    },
                },
                "required": ["type"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the training run",
            },
            "log_level": {
                "description": "Optional logging level (int or string, e.g. INFO)",
                "anyOf": [{"type": "integer"}, {"type": "string"}],
            },
            "training": {
                "type": "object",
                "description": "Training configuration",
                "properties": {
                    "seed": {"type": "integer", "description": "Random seed"},
                    "device": {
                        "type": "string",
                        "description": "Torch device string, e.g. 'cpu', 'cuda', 'cuda:0'",
                    },
                    "path": {
                        "type": "string",
                        "description": "Output directory for run artifacts and checkpoints",
                    },
                    "num_epochs": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of training epochs",
                    },
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Training DataLoader batch size",
                    },
                    "optimizer_type": {
                        "description": "Optimizer type name, e.g. 'torch.optim.Adam' or 'torch.optim.SGD'",
                    },
                    "optimizer_args": {
                        "type": "array",
                        "description": "Arguments for optimizer",
                        "items": {},
                    },
                    "optimizer_kwargs": {
                        "type": "object",
                        "description": "Optimizer keyword arguments",
                        "additionalProperties": {},
                    },
                    "lr_scheduler_type": {
                        "description": "type of the learning rate scheduler", 
                    },
                    "lr_scheduler_args": {
                        "type": "array",
                        "description": "arguments to construct the learning rate scheduler", 
                        "items": {},
                    },
                    "lr_scheduler_kwargs": {
                        "type": "object",
                        "description": "keyword arguments for the construction of learning rate scheduler", 
                        "additionalProperties": {},
                    },
                    "num_workers": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "DataLoader workers for training",
                    },
                    "pin_memory": {
                        "type": "boolean",
                        "description": "Pin GPU memory in DataLoader",
                    },
                    "drop_last": {
                        "type": "boolean",
                        "description": "Drop last incomplete batch",
                    },
                    "prefetch_factor": {
                        "type": ["integer", "null"],
                        "minimum": 2,
                        "description": "Prefetch samples per worker (None or >=2)",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Shuffle training dataset",
                    },
                    "checkpoint_at": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Checkpoint every N epochs (or None to disable)",
                    },
                },
                "required": [
                    "seed",
                    "device",
                    "path",
                    "num_epochs",
                    "batch_size",
                    "optimizer_type",
                    "optimizer_args",
                    "optimizer_kwargs",
                    "num_workers",
                    "drop_last",
                    "checkpoint_at",
                ],
                "additionalProperties": True,
            },
            "data": {
                "type": "object",
                "description": "Dataset configuration",
                "properties": {
                    "pre_transform": {
                        "description": "Name of the python object to use for the pre-transform function to use. Must refer to a callable"
                    },
                    "transform": {
                        "description": "Name of the python object to use for the transform function to use. Must refer to a callable"
                    },
                    "pre_filter": {
                        "description": "Name of the python object to use for the pre_filter function to use. Must refer to a callable"
                    },
                    "reader": {
                        "description": "Name  of the python object to read raw data from file. Must be callable",
                    },
                    "files": {
                        "type": "array",
                        "description": "list of zarr stores to get data from",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "description": "zarr file names to read data from",
                        },
                    },
                    "output": {
                        "type": "string",
                        "description": "path to store preprocessed data at.",
                    },
                    "validate_data": {
                        "type": "boolean",
                        "description": "Whether to validate the transformed data objects or not",
                    },
                    "n_processes": {
                        "type": "integer",
                        "description": "number of processes to use for preprocessing the dataset",
                        "minimum": 0,
                    },
                    "chunksize": {
                        "type": "integer",
                        "description": "Number of datapoints to process at once during preprocessing",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Whether to shuffle the dataset or not",
                    },
                    "subset": {
                        "type": "number",
                        "description": "Fraction of the dataset to use. Full dataset is used when not given",
                    },
                    "split": {
                        "type": "array",
                        "description": "Split ratios of the dataset",
                        "items": {
                            "type": "number",
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                },
                "required": ["output", "files", "reader"],
                "additionalProperties": False,
            },
            "model": {
                "description": "Model config: either constructor triple or full GNNModel schema",
                "anyOf": [
                    {"$ref": "#/definitions/constructor"},
                    gnn_model.GNNModel.schema,
                ],
            },
            "validation": {
                "type": "object",
                "description": "Model validation configuration",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Validation DataLoader batch size",
                    },
                    "num_workers": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "DataLoader workers",
                    },
                    "pin_memory": {
                        "type": "boolean",
                        "description": "Pin GPU memory in DataLoader",
                    },
                    "drop_last": {
                        "type": "boolean",
                        "description": "Drop last incomplete batch",
                    },
                    "prefetch_factor": {
                        "type": ["integer", "null"],
                        "minimum": 2,
                        "description": "Prefetch samples per worker (None or >=2)",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Shuffle validation dataset",
                    },
                    "validator": {
                        "$ref": "#/definitions/constructor",
                        "description": "Validator constructor spec: provides type, args, kwargs",
                    },
                },
                "required": ["batch_size"],
                "additionalProperties": True,
            },
            "testing": {
                "type": "object",
                "description": "Configuration for model testing after training",
                "properties": {
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Test DataLoader batch size",
                    },
                    "num_workers": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "DataLoader workers",
                    },
                    "pin_memory": {
                        "type": "boolean",
                        "description": "Pin GPU memory in DataLoader",
                    },
                    "drop_last": {
                        "type": "boolean",
                        "description": "Drop last incomplete batch",
                    },
                    "prefetch_factor": {
                        "type": ["integer", "null"],
                        "minimum": 2,
                        "description": "Prefetch samples per worker (None or >=2)",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Shuffle test dataset",
                    },
                    "tester": {
                        "$ref": "#/definitions/constructor",
                        "description": "Tester constructor spec: provides type, args, kwargs",
                    },
                },
                "required": ["batch_size"],
                "additionalProperties": True,
            },
            "early_stopping": {
                "$ref": "#/definitions/constructor",
                "description": "Early stopping constructor spec: provides type, args, kwargs",
            },
            "apply_model": {
                "description": "Optional method to call the model on data. Useful when using optional signatures for instance "
            },
            "criterion": {
                "description": "The loss function used for training as a python type"
            },
        },
        "required": [
            "training",
            "model",
            "validation",
            "testing",
            "criterion",
        ],
        "additionalProperties": True,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        # training and evaluation functions
    ):
        """Initialize the trainer.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """

        jsonschema.validate(instance=config, schema=self.schema)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.get("log_level", logging.INFO))
        self.logger.info("Initializing Trainer instance")

        # functions for executing training and evaluation
        self.criterion = config["criterion"]
        self.apply_model = config.get("apply_model")
        self.seed = config["training"]["seed"]
        self.device = torch.device(config["training"]["device"])
        
        self.nprng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # parameters for finding out which model is best
        self.best_score = None
        self.best_epoch = 0
        self.epoch = 0

        # date and time of run:
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.data_path = (
            Path(self.config["training"]["path"])
            / f"{config.get('name', 'run')}_{run_date}"
        )

        # set up paths for storing model snapshots and data
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)
        self.logger.info(f"Data path set to: {self.data_path}")

        self.checkpoint_path = self.data_path / "model_checkpoints"
        self.checkpoint_at = config["training"].get("checkpoint_at", None)
        self.latest_checkpoint = None

        # model and optimizer initialization placeholders
        self.model = None
        self.optimizer = None

        # early stopping and evaluation functors
        try:
            self.early_stopping = early_stopping.DefaultEarlyStopping.from_config(
                config["early_stopping"]
            )
        except Exception as e:
            self.logger.debug(
                f"from_config failed for early stopping, using direct instantiation: {e}"
            )
            self.early_stopping = config["early_stopping"]["type"](
                *config["early_stopping"]["args"], **config["early_stopping"]["kwargs"]
            )

        try:
            self.validator = evaluate.DefaultValidator.from_config(
                config["validation"]["validator"]
            )
        except Exception as e:
            self.logger.debug(
                f"from_config failed for validator, using direct instantiation: {e}"
            )
            self.validator = config["validation"]["validator"]["type"](
                *config["validation"]["validator"]["args"],
                **config["validation"]["validator"]["kwargs"],
            )

        try:
            self.tester = evaluate.DefaultTester.from_config(
                config["testing"]["tester"]
            )
        except Exception as e:
            self.logger.debug(
                f"from_config failed for tester, using direct instantiation: {e}"
            )
            self.tester = config["testing"]["tester"]["type"](
                *config["testing"]["tester"]["args"],
                **config["testing"]["tester"]["kwargs"],
            )

        with open(self.data_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        self.logger.info("Trainer initialized")
        self.logger.debug(f"Configuration: {self.config}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Trainer":
        """Create a Trainer instance from a configuration dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        return cls(
            config=config,
        )

    def initialize_model(self) -> Any:
        """Initialize the model for training.

        Returns:
            Any: The initialized model.
        """
        if hasattr(self, "model") and self.model is not None:
            self.logger.warning(
                "Model is already initialized. This will replace it with a new instance"
            )

        try:
            self.model = gnn_model.GNNModel.from_config(self.config["model"]).to(
                self.device
            )

        except Exception:
            self.logger.debug(
                "from_config for  model initialization failed, using direct initialization instead"
            )
            self.model = self.config["model"]["type"](
                *self.config["model"]["args"], **self.config["model"]["kwargs"]
            ).to(self.device)

        self.logger.info("Model initialized to device: {}".format(self.device))
        return self.model

    def initialize_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Initialize the learning rate scheduler for training.

        Raises:
            RuntimeError: If the optimizer is not initialized.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The initialized learning rate scheduler.
        """
        if self.config["training"].get("lr_scheduler_type") is None:
            self.logger.info("No learning rate scheduler specified in config.")
            return None
        else:
            if not hasattr(self, "optimizer") or self.optimizer is None:
                raise RuntimeError(
                    "Optimizer must be initialized before initializing learning rate scheduler."
                )
            
            try:
                self.lr_scheduler = self.config["training"].get("lr_scheduler_type")(
                    self.optimizer,
                    *self.config["training"].get("lr_scheduler_args", []),
                    **self.config["training"].get("lr_scheduler_kwargs", {}),
                )
                self.logger.info("Learning rate scheduler initialized.")
                return self.lr_scheduler
            except Exception as e:
                self.logger.error(f"Error initializing learning rate scheduler: {e}")
                raise e

    def initialize_optimizer(self) -> torch.optim.Optimizer | None:
        """Initialize the optimizer for training.

        Raises:
            RuntimeError: If the model is not initialized.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """

        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model must be initialized before initializing optimizer."
            )

        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.logger.warning(
                "Optimizer is already initialized. This will replace it with a new instance"
            )

        try:
            optimizer = self.config["training"].get("optimizer_type", torch.optim.Adam)(
                self.model.parameters(),
                *self.config["training"].get("optimizer_args", []),
                **self.config["training"].get("optimizer_kwargs", {}),
            )
            self.optimizer = optimizer
            self.logger.info("Optimizer initialized")
        except Exception as e:
            self.logger.error(f"Error initializing optimizer: {e}")
            raise e
        return self.optimizer

    def prepare_dataset(
        self,
        dataset: Dataset | None = None,
        split: list[float] = [0.8, 0.1, 0.1],
        train_dataset: torch.utils.data.Subset | None = None,
        val_dataset: torch.utils.data.Subset | None = None,
        test_dataset: torch.utils.data.Subset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Set up the split for training, validation, and testing datasets.

        Args:
            dataset (Dataset | None, optional): Dataset to be split. Only one of dataset, train_dataset, val_dataset, test_dataset should be provided. Defaults to None.
            split (list[float], optional): split ratios for train, validation, and test datasets. Defaults to [0.8, 0.1, 0.1].
            train_dataset (torch.utils.data.Subset | None, optional): Training subset of the dataset. Only one of dataset, train_dataset, val_dataset, test_dataset should be provided. Defaults to None.
            val_dataset (torch.utils.data.Subset | None, optional): Validation subset of the dataset. Only one of dataset, train_dataset, val_dataset, test_dataset should be provided. Defaults to None.
            test_dataset (torch.utils.data.Subset | None, optional): Testing subset of the dataset. Only one of dataset, train_dataset, val_dataset, test_dataset should be provided. Defaults to None.

        Raises:
            ValueError: If providing train, val, or test datasets, the full dataset must not be provided.
            ValueError: If split ratios are not summing up to 1
            ValueError: If train size is 0
            ValueError: If validation size is 0
            ValueError: If test size is 0

        Returns:
            Tuple[Dataset, Dataset, Dataset]: train, validation, and test datasets.
        """
        if dataset is not None and (
            train_dataset is not None
            or val_dataset is not None
            or test_dataset is not None
        ):
            raise ValueError(
                "If providing train, val, or test datasets, the full dataset must not be provided."
            )

        if dataset is None:
            cfg = self.config["data"]
            dataset = dataset_ondisk.QGDataset(
                cfg["files"],
                cfg["output"],
                cfg["reader"],
                float_type=cfg.get("float_type", torch.float32),
                int_type=cfg.get("int_type", torch.int32),
                validate_data=cfg.get("validate_data", True),
                chunksize=cfg.get("chunksize", 1),
                n_processes=cfg.get("n_processes", 1),
                transform=cfg.get("transform"),
                pre_transform=cfg.get("pre_transform"),
                pre_filter=cfg.get("pre_filter"),
            )

            if cfg.get("subset"):
                num_points = ceil(len(dataset) * cfg["subset"])
                dataset = dataset.index_select(
                    self.nprng.integers(0, len(dataset),size=num_points).tolist()
                )

            if cfg.get("shuffle"):
                dataset.shuffle()

        if train_dataset is None and val_dataset is None and test_dataset is None:
            split = self.config.get("data", {}).get("split", split)
            if not np.isclose(
                np.sum(split), 1.0, rtol=1e-05, atol=1e-08, equal_nan=False
            ):
                raise ValueError(
                    f"Split ratios must sum to 1.0. Provided split: {split}"
                )

            train_size = ceil(len(dataset) * split[0])
            val_size = floor(len(dataset) * split[1])
            test_size = len(dataset) - train_size - val_size

            if train_size == 0:
                raise ValueError("train size cannot be 0")

            if val_size == 0:
                raise ValueError("validation size cannot be 0")

            if test_size == 0:
                raise ValueError("test size cannot be 0")

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

        return train_dataset, val_dataset, test_dataset

    def prepare_dataloaders(
        self,
        dataset: Dataset | None = None,
        split: list[float] = [0.8, 0.1, 0.1],
        train_dataset: torch.utils.data.Subset | None = None,
        val_dataset: torch.utils.data.Subset | None = None,
        test_dataset: torch.utils.data.Subset | None = None,
        training_sampler: torch.utils.data.Sampler | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare the data loaders for training, validation, and testing.

        Args:
            dataset (Dataset): The dataset to prepare.
            split (list[float], optional): The split ratios for training, validation, and test sets. Defaults to [0.8, 0.1, 0.1].
            training_sampler (torch.utils.data.Sampler, optional): The sampler for the training data loader. Defaults to None.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The data loaders for training, validation, and testing.
        """
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_dataset(
            dataset=dataset,
            split=split,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
        train_loader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=self.config["training"].get("pin_memory", True),
            drop_last=self.config["training"].get("drop_last", False),
            prefetch_factor=self.config["training"].get("prefetch_factor", None),
            shuffle=self.config["training"].get("shuffle", True),
            sampler=training_sampler,
        )

        val_loader = DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.config["validation"]["batch_size"],
            num_workers=self.config["validation"].get("num_workers", 0),
            pin_memory=self.config["validation"].get("pin_memory", True),
            drop_last=self.config["validation"].get("drop_last", False),
            prefetch_factor=self.config["validation"].get("prefetch_factor", None),
            shuffle=self.config["validation"].get("shuffle", True),
        )

        test_loader = DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.config["testing"]["batch_size"],
            num_workers=self.config["testing"].get("num_workers", 0),
            pin_memory=self.config["testing"].get("pin_memory", True),
            drop_last=self.config["testing"].get("drop_last", False),
            prefetch_factor=self.config["testing"].get("prefetch_factor", None),
            shuffle=self.config["testing"].get("shuffle", True),
        )

        if dataset is not None:
            self.logger.info(
                f"Data loaders prepared with splits: {split} and dataset sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}"
            )
        return train_loader, val_loader, test_loader

    # training helper functions
    def _evaluate_batch(
        self,
        model: torch.nn.Module,
        data: Data,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """Evaluate a single batch of data using the model.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data (Data): The input data for the model.

        Returns:
            torch.Tensor | Collection[torch.Tensor]: The output of the model.
        """
        self.logger.debug(f"  Evaluating batch on device: {self.device}")
        if self.apply_model:
            outputs = self.apply_model(model, data)
        else:
            outputs = model(data.x, data.edge_index, data.batch)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outputs

    def _run_train_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
    ) -> torch.Tensor:
        """Run a single training epoch.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            train_loader (DataLoader): The data loader for the training set.
        Raises:
            RuntimeError: If the model is not initialized.
            RuntimeError: If the optimizer is not initialized.

        Returns:
            torch.Tensor: The training loss for each batch stored in a torch.Tensor
        """

        if model is None:
            raise RuntimeError("Model must be initialized before training.")

        if optimizer is None:
            raise RuntimeError("Optimizer must be initialized before training.")

        losses = torch.zeros(len(train_loader), dtype=torch.float32, device=self.device)
        self.logger.info(f"  Starting training epoch {self.epoch}")
        # training run
        for i, batch in enumerate(
            tqdm.tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        ):
            self.logger.debug(f"    Moving batch {i} to device: {self.device}")
            optimizer.zero_grad()

            data = batch.to(self.device)
            outputs = self._evaluate_batch(model, data)

            self.logger.debug("    Computing loss")
            loss = self.criterion(outputs, data, self)

            self.logger.debug(f"    Backpropagating loss: {loss.item()}")
            loss.backward()

            optimizer.step()

            losses[i] = loss

        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return losses

    def _check_model_status(self, eval_data: pd.DataFrame) -> bool:
        """Check the status of the model during training.

        Args:
            eval_data (pd.DataFrame): The evaluation data from the training epoch.

        Returns:
            bool: Whether the training should stop early.
        """
        if self.model is None: 
            raise ValueError("Model must be initialized before saving checkpoints")
        
        if (
            self.checkpoint_at is not None
            and self.epoch % self.checkpoint_at == 0
            and self.epoch > 0
        ):
            self.save_checkpoint()

        if self.early_stopping is not None:
            if self.early_stopping(eval_data):
                self.logger.debug(f"Early stopping at epoch {self.epoch}.")
                self.save_checkpoint(name_addition=f"_{self.epoch}_early_stopping")
                return True

            if self.early_stopping.found_better_model:
                self.logger.debug(f"Found better model at epoch {self.epoch}.")
                self.save_checkpoint(name_addition=f"_{self.epoch}_current_best")
                # not returning true because this is not the end of training

        return False

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: optuna.trial.Trial | None = None,
    ) -> Tuple[torch.Tensor | Collection[Any], torch.Tensor | Collection[Any]]:
        """Run the training process.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            trial (optuna.trial.Trial | None, optional): An Optuna trial
                for hyperparameter tuning. Defaults to None.

        Returns:
            Tuple[torch.Tensor | Collection[Any], torch.Tensor | Collection[Any]]: The training and validation results.
        """
        self.logger.info("Starting training process.")
        # training loop
        num_epochs = self.config["training"]["num_epochs"]

        self.initialize_model()

        self.initialize_optimizer()

        self.initialize_lr_scheduler()

        total_training_data = torch.zeros(num_epochs, 2, dtype=torch.float32)

        for epoch in range(0, num_epochs):
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")
            self.model.train()

            epoch_data = self._run_train_epoch(self.model, self.optimizer, train_loader)

            # collect mean and std for each epoch
            total_training_data[epoch, :] = torch.Tensor(
                [epoch_data.mean(dim=0).item(), epoch_data.std(dim=0).item()]
            )

            self.logger.info(
                f"  Completed epoch {epoch}. training loss: {total_training_data[epoch, 0]:.8f} +/- {total_training_data[epoch, 1]:.8f}."
            )

            # evaluation run on validation set
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                self.validator.report(validation_result)

                # integrate Optuna here for hyperparameter tuning
                if trial is not None:
                    avg_sigma_loss = self.validator.data[self.epoch]
                    avg_loss = avg_sigma_loss[0]
                    trial.report(avg_loss, self.epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            should_stop = self._check_model_status(
                self.validator.data if self.validator else total_training_data,
            )
            if should_stop:
                self.logger.info("Stopping training early.")
                break
            self.epoch += 1

        self.logger.info("Training process completed.")
        self.logger.info("Saving model")

        outpath = self.data_path / f"final_model_epoch={self.epoch}.pt"
        self.model.save(outpath)

        return total_training_data, self.validator.data if self.validator else []

    def run_test(
        self, test_loader: DataLoader, model_name_addition: str = "current_best.pt"
    ) -> Collection[Any]:
        """Run testing phase.

        Args:
            test_loader (DataLoader): The data loader for the test set.
            model_name_addition (str): An optional string to append to the checkpoint filename.
        Raises:
            RuntimeError: If the model is not initialized.
            RuntimeError: If the test data is not available.

        Returns:
            Collection[Any]: A collection of test results that can be scalars, tensors, lists, dictionaries or any other data type that the tester might return.
        """
        self.logger.info("Starting testing process.")
        # get the best model again

        saved_models = [
            f
            for f in Path(self.checkpoint_path).iterdir()
            if f.is_file() and model_name_addition in str(f)
        ]

        if len(saved_models) == 0:
            raise RuntimeError(
                f"No model with the name addition '{model_name_addition}' found, did training work?"
            )

        # get the latest of the best models
        best_of_the_best = max(saved_models, key=lambda f: f.stat().st_mtime)

        self.logger.info(f"loading best model found: {str(best_of_the_best)}")

        self.model = gnn_model.GNNModel.load(
            best_of_the_best, self.config["model"], device=self.device
        )
        self.model.eval()
        if self.tester is None:
            raise RuntimeError("Tester must be initialized before testing.")
        test_result = self.tester.test(self.model, test_loader)
        self.tester.report(test_result)
        self.logger.info("Testing process completed.")
        self.save_checkpoint(name_addition="best_model_found")
        return self.tester.data

    def save_checkpoint(self, name_addition: str = ""):
        """Save model checkpoint.

        Raises:
            ValueError: If the model is not initialized.
            ValueError: If the model configuration does not contain 'name'.
            ValueError: If the training configuration does not contain 'checkpoint_path'.
        """
        self.logger.info(
            f"Saving checkpoint for model at epoch {self.epoch} to {self.checkpoint_path}"
        )
        outpath = self.checkpoint_path / f"model_{name_addition}.pt"

        if outpath.exists() is False:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory {outpath.parent} for checkpoint.")

        self.latest_checkpoint = outpath
        self.model.save(outpath)

    def load_checkpoint(self, name_addition: str = "") -> None:
        """Load model checkpoint to the device given

        Args:
            name_addition (str): An optional string to append to the checkpoint filename.

        Raises:
            RuntimeError: If the model is not initialized.
        """

        if self.model is None:
            raise RuntimeError("Model must be initialized before loading checkpoint.")

        if Path(self.checkpoint_path).exists() is False:
            raise RuntimeError("Checkpoint path does not exist.")

        self.logger.info(
            "available checkpoints: %s", list(Path(self.checkpoint_path).iterdir())
        )

        loadpath = Path(self.checkpoint_path) / f"model_{name_addition}.pt"

        if not loadpath.exists():
            raise FileNotFoundError(f"Checkpoint file {loadpath} does not exist.")

        self.model = gnn_model.GNNModel.load(
            loadpath,
            self.config["model"],
        )
