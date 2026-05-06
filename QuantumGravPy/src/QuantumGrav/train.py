from collections.abc import Collection
from typing import Any, Callable, Tuple

from pathlib import Path
import logging
import tqdm
import yaml
from datetime import datetime
import jsonschema
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass

from . import evaluate
from . import early_stopping
from . import gnn_model
from . import base
from .config_utils import get_loader, convert_to_pyobject_tags
from .utils import seed_all_rngs

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import optuna


@dataclass
class Snapshot:
    epoch: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    lr_scheduler_state_dict: dict[str, Any] | None
    config_path: Path | str
    path: Path | str
    early_stopping_state_dict: dict[str, Any] | None = None
    validator_state_dict: dict[str, Any] | None = None
    tester_state_dict: dict[str, Any] | None = None

    @classmethod
    def from_trainer(cls, trainer: "Trainer") -> "Snapshot":
        """Create a Snapshot instance from a Trainer instance.

        Args:
            trainer (Trainer): The Trainer instance to create the snapshot from.
        """

        return cls(
            epoch=trainer.epoch,
            model_state_dict=trainer.model.state_dict() if trainer.model else None,
            optimizer_state_dict=trainer.optimizer.state_dict()
            if trainer.optimizer
            else None,
            lr_scheduler_state_dict=trainer.lr_scheduler.state_dict()
            if trainer.lr_scheduler
            else None,
            config_path=trainer.data_path / "config.yaml",
            path=Path(trainer.checkpoint_path) / f"epoch_{trainer.epoch}",
            validator_state_dict=trainer.validator.to_state_dict()
            if trainer.validator
            else None,
            tester_state_dict=trainer.tester.to_state_dict()
            if trainer.tester
            else None,
            early_stopping_state_dict=trainer.early_stopper.to_state_dict()
            if trainer.early_stopper
            else None,
        )

    def save(self):
        """Save the snapshot to disk."""
        self.path = Path(self.path)
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                "lr_scheduler_state_dict": self.lr_scheduler_state_dict,
                "config_path": str(self.config_path),
                "validator_state_dict": self.validator_state_dict,
                "tester_state_dict": self.tester_state_dict,
                "early_stopping_state_dict": self.early_stopping_state_dict,
            },
            self.path,
        )

    @classmethod
    def load(cls, load_path: Path | str) -> "Snapshot":
        """Load a snapshot from disk.

        Args:
            load_path (Path | str): The path to the snapshot file.

        Returns:
            Snapshot: The loaded Snapshot instance.
        """
        checkpoint = torch.load(load_path, weights_only=False)

        if not all(
            key in checkpoint
            for key in [
                "epoch",
                "model_state_dict",
                "optimizer_state_dict",
                "config_path",
            ]
        ):
            raise ValueError(
                f"Checkpoint file {load_path} is missing required keys. Found keys: {
                    list(checkpoint.keys())
                }. Needs keys: {
                    [
                        'epoch',
                        'model_state_dict',
                        'optimizer_state_dict',
                        'config_path',
                    ]
                }"
            )

        return Snapshot(
            epoch=checkpoint["epoch"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            lr_scheduler_state_dict=checkpoint.get("lr_scheduler_state_dict", None),
            config_path=checkpoint["config_path"],
            path=load_path,
            validator_state_dict=checkpoint.get("validator_state_dict", None),
            tester_state_dict=checkpoint.get("tester_state_dict", None),
            early_stopping_state_dict=checkpoint.get("early_stopping_state_dict", None),
        )


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
                        "description": "Validator configuration: either constructor spec or Evaluator schema",
                        "anyOf": [
                            {"$ref": "#/definitions/constructor"},
                            evaluate.Evaluator.schema,
                        ],
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
                        "description": "Tester configuration: either constructor spec or Evaluator schema",
                        "anyOf": [
                            {"$ref": "#/definitions/constructor"},
                            evaluate.Evaluator.schema,
                        ],
                    },
                },
                "required": ["batch_size"],
                "additionalProperties": True,
            },
            "early_stopping": {
                "description": "Early stopping configuration: either constructor spec or DefaultEarlyStopping schema",
                "anyOf": [
                    {"$ref": "#/definitions/constructor"},
                    early_stopping.DefaultEarlyStopping.schema,
                ],
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
        config: dict[str, Any],
        logger: logging.Logger,
        criterion: Callable,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        seed: int,
        device: torch.device,
        data_path: Path,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        early_stopper: early_stopping.DefaultEarlyStopping | None = None,
        validator: evaluate.Evaluator | None = None,
        tester: evaluate.Evaluator | None = None,
        apply_model: Callable | None = None,
    ):
        self.config = config
        self.logger = logger
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.apply_model = apply_model
        self.validator = validator
        self.early_stopper = early_stopper
        self.tester = tester
        self.seed = seed
        self.device = device
        self.epoch = 0
        self.data_path = data_path
        self.checkpoint_path = data_path / "checkpoints"
        self.checkpoint_at = config["training"].get("checkpoint_at", None)

        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        if model is None:
            self.initialize_model()

        if optimizer is None:
            self.initialize_optimizer()

        if lr_scheduler is None:
            self.initialize_lr_scheduler()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Trainer":
        """Create a Trainer instance from a configuration dictionary.

        Args:
            config (dict[str, Any]): The configuration dictionary.
        """
        jsonschema.validate(instance=config, schema=cls.schema)

        config = config
        logger = logging.getLogger(__name__)
        logger.setLevel(config.get("log_level", logging.INFO))
        logger.info("Initializing Trainer instance")

        # functions for executing training and evaluation
        criterion = config["criterion"]
        apply_model = config.get("apply_model")
        seed = config["training"]["seed"]
        device = torch.device(config["training"]["device"])

        seed_all_rngs(seed)

        # date and time of run:
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_path = (
            Path(config["training"]["path"]) / f"{config.get('name', 'run')}_{run_date}"
        )

        # set up paths for storing model snapshots and data
        if not data_path.exists():
            data_path.mkdir(parents=True)
        logger.info(f"Data path set to: {data_path}")

        # early stopping and evaluation functors
        if "early_stopping" in config:
            early_stopping_cfg = deepcopy(config["early_stopping"])
            try:
                early_stopper = early_stopping.DefaultEarlyStopping.from_config(
                    early_stopping_cfg
                )
            except Exception as e:
                logger.debug(
                    f"from_config failed for early stopping, using direct instantiation: {e}"
                )

                early_stopper = early_stopping_cfg["type"](
                    *early_stopping_cfg["args"],
                    **early_stopping_cfg["kwargs"],
                )
        else:
            early_stopper = None

        if "validation" in config:
            try:
                validator = evaluate.Validator.from_config(
                    config["validation"]["validator"]
                )
            except Exception as e:
                logger.debug(
                    f"from_config failed for validator, using direct instantiation: {e}"
                )
                validator = config["validation"]["validator"]["type"](
                    *config["validation"]["validator"]["args"],
                    **config["validation"]["validator"]["kwargs"],
                )
        else:
            validator = None

        if "testing" in config:
            try:
                tester = evaluate.Tester.from_config(config["testing"]["tester"])
            except Exception as e:
                logger.debug(
                    f"from_config failed for tester, using direct instantiation: {e}"
                )
                tester = config["testing"]["tester"]["type"](
                    *config["testing"]["tester"]["args"],
                    **config["testing"]["tester"]["kwargs"],
                )
        else:
            tester = None

        with open(data_path / "config.yaml", "w") as f:
            yaml.safe_dump(
                convert_to_pyobject_tags(config, emit_yaml_tags=True),
                f,
                sort_keys=False,
            )

        trainer = cls(
            config=config,
            logger=logger,
            criterion=criterion,
            model=None,
            optimizer=None,
            lr_scheduler=None,
            apply_model=apply_model,
            validator=validator,
            early_stopper=early_stopper,
            tester=tester,
            seed=seed,
            device=device,
            data_path=data_path,
        )

        logger.info("Trainer initialized")
        logger.debug(f"Configuration: {config}")

        return trainer

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
            try:
                self.model = self.config["model"]["type"](
                    *self.config["model"].get("args", []),
                    **self.config["model"].get("kwargs", {}),
                ).to(self.device)
            except Exception as e:
                self.logger.error(f"Error initializing model from constructor: {e}")
                raise e

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

        losses = torch.zeros(
            len(train_loader),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
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

            losses[i] = loss.detach().clone()

        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return losses

    def _run_validation_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loader: DataLoader,
        trial: optuna.trial.Trial | None = None,
    ) -> None:
        """_summary_

        Args:
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            val_loader (DataLoader): _description_
            trial (optuna.trial.Trial | None, optional): An Optuna trial for hyperparameter tuning. Defaults to None.

        """
        if self.validator is None:
            self.logger.info("No validator specified, skipping validation epoch.")
            return

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

        if self.early_stopper is not None:
            if self.early_stopper(eval_data):
                self.logger.debug(f"Early stopping at epoch {self.epoch}.")
                return True

            if self.early_stopper.found_better_model:
                self.logger.debug(f"Saving better model at epoch {self.epoch}.")
                # not returning true because this is not the end of training

        return False

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: optuna.trial.Trial | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the training process.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.
            trial (optuna.trial.Trial | None, optional): An Optuna trial
                for hyperparameter tuning. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The training and validation results.
        """
        self.logger.info("Starting training process.")
        # training loop
        num_epochs = self.config["training"]["num_epochs"]

        if self.model is None:
            self.initialize_model()

        if self.optimizer is None:
            self.initialize_optimizer()

        if self.lr_scheduler is None:
            self.initialize_lr_scheduler()

        total_training_data = pd.DataFrame(
            columns=[
                "epoch",
                "min_loss",
                "mean_loss",
                "max_loss",
                "median_loss",
                "std_loss",
            ],
        )

        while self.epoch < num_epochs:
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")
            self.model.train()

            epoch_data = self._run_train_epoch(self.model, self.optimizer, train_loader)

            # collect mean and std for each epoch
            total_training_data.loc[self.epoch] = {
                "epoch": self.epoch,
                "min_loss": epoch_data.min().item(),
                "mean_loss": epoch_data.mean().item(),
                "max_loss": epoch_data.max().item(),
                "median_loss": epoch_data.median().item(),
                "std_loss": epoch_data.std().item(),
            }
            self.logger.info(
                f"  Completed epoch {self.epoch} \n {total_training_data.tail(1).to_string()}"
            )

            # evaluation run on validation set
            if self.validator is not None:
                self.model.eval()
                self._run_validation_epoch(
                    self.model, self.optimizer, val_loader, trial
                )

            should_stop = self._check_model_status(
                self.validator.data if self.validator else total_training_data,
            )
            if should_stop:
                self.logger.info("Stopping training early.")
                self.save_checkpoint(name_addition="_final")
                break

            if self.early_stopper is not None and self.early_stopper.found_better_model:
                self.save_checkpoint(name_addition="_current_best")
            self.epoch += 1  # this means that the epoch number in the checkpoint will be the last completed epoch, which is intuitive for resuming training from checkpoints

        self.logger.info("Training process completed.")
        self.logger.info("Saving model")

        outpath = self.checkpoint_path / f"final_model_epoch={self.epoch}"

        snapshot = Snapshot.from_trainer(self)
        snapshot.path = outpath

        try:
            snapshot.save()
        except Exception as e:
            self.logger.error(f"Error saving final model: {e}")
            raise e

        return (
            total_training_data,
            self.validator.data if self.validator else pd.DataFrame(),
        )

    def run_test(
        self, test_loader: DataLoader, model_name_addition: str = "_current_best"
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
        if self.tester is None:
            self.logger.info("No tester specified, skipping testing phase.")
            return {}

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

        # load model:
        self.model = self.load_model(best_of_the_best)

        self.model.eval()
        if self.tester is None:
            raise RuntimeError("Tester must be initialized before testing.")
        test_result = self.tester.test(self.model, test_loader)
        self.tester.report(test_result)
        self.logger.info("Testing process completed.")
        self.save_checkpoint(name_addition="final_tested_model")
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

        snapshot = Snapshot.from_trainer(self)
        if name_addition:
            snapshot.path = self.checkpoint_path / f"epoch_{self.epoch}{name_addition}"

        try:
            snapshot.save()
            self.logger.info(f"Checkpoint saved to {snapshot.path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise e

    def load_model(self, model_path: str | Path) -> torch.nn.Module:
        """Load a model from a checkpoint.

        Args:
            model_path (str | Path): The path to the model checkpoint.

        Returns:
            torch.nn.Module: The loaded model.
        """

        if self.model is not None:
            self.logger.warning(
                "Model is already initialized. This will replace it with the loaded model from checkpoint."
            )

        try:
            self.model = gnn_model.GNNModel.from_config(self.config["model"]).to(
                self.device
            )
            snapshot = Snapshot.load(model_path)
            state_dict = snapshot.model_state_dict
            self.model.load_state_dict(state_dict)

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise e

        self.logger.info("Model loaded from checkpoint: {}".format(model_path))

        return self.model

    @classmethod
    def load_checkpoint(
        cls, load_path: Path | str, new_path: Path | str | None = None
    ) -> "Trainer":
        """Load model checkpoint to the device given

        Args:
            load_path (Path | str): The path to the checkpoint file.
            new_path (Path | str | None): The path to the new checkpoint file.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        path = Path(load_path)
        # load snapshot
        snapshot = Snapshot.load(path)

        with open(snapshot.config_path) as f:
            config = yaml.load(f, Loader=get_loader())

        trainer = cls.from_config(config)

        # initialize model
        if trainer.model is not None:
            trainer.model.load_state_dict(snapshot.model_state_dict)
        else:
            raise RuntimeError("Model must be initialized before loading checkpoint.")

        # initialize optimizer
        if trainer.optimizer is not None and snapshot.optimizer_state_dict is not None:
            trainer.optimizer.load_state_dict(snapshot.optimizer_state_dict)
        else:
            raise RuntimeError(
                "Optimizer must be initialized before loading checkpoint."
            )

        # initialize lr scheduler
        if (
            trainer.lr_scheduler is not None
            and snapshot.lr_scheduler_state_dict is not None
        ):
            trainer.lr_scheduler.load_state_dict(snapshot.lr_scheduler_state_dict)

        # set data for validator
        if trainer.validator is not None and snapshot.validator_state_dict is not None:
            trainer.validator.load_state_dict(snapshot.validator_state_dict)

        if trainer.tester is not None and snapshot.tester_state_dict is not None:
            trainer.tester.load_state_dict(snapshot.tester_state_dict)

        if (
            trainer.early_stopper is not None
            and snapshot.early_stopping_state_dict is not None
        ):
            trainer.early_stopper.load_state_dict(snapshot.early_stopping_state_dict)

        # set epoch
        trainer.epoch = snapshot.epoch
        return trainer
