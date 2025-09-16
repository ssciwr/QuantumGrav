from collections.abc import Collection
from typing import Callable, Any, Tuple

import numpy as np
from pathlib import Path
import logging
import tqdm
import yaml
from datetime import datetime

from .evaluate import DefaultValidator, DefaultTester
from . import gnn_model

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import optuna


class Trainer:
    """Trainer class for training and evaluating GNN models."""

    def __init__(
        self,
        config: dict[str, Any],
        # training and evaluation functions
        criterion: Callable[[Any, Data], torch.Tensor],
        apply_model: Callable | None = None,
        # training evaluation and reporting
        early_stopping: Callable[[Collection[Any] | torch.Tensor], bool] | None = None,
        validator: DefaultValidator | None = None,
        tester: DefaultTester | None = None,
    ):
        """Initialize the trainer.

        Args:
            config (dict[str, Any]): The configuration dictionary.
            criterion (Callable): The loss function to use.
            apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
            early_stopping (Callable[[Collection[Any]], bool] | None, optional): A function for early stopping. Defaults to None.
            validator (DefaultValidator | None, optional): A validator for model evaluation. Defaults to None.
            tester (DefaultTester | None, optional): A tester for model evaluation. Defaults to None.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if (
            all(x in config for x in ["training", "model", "validation", "testing"])
            is False
        ):
            raise ValueError(
                "Configuration must contain 'training', 'model', 'validation' and 'testing' sections."
            )

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.get("log_level", logging.INFO))
        self.logger.info("Initializing Trainer instance")

        # functions for executing training and evaluation
        self.criterion = criterion
        self.apply_model = apply_model
        self.early_stopping = early_stopping
        self.seed = config["training"]["seed"]
        self.device = torch.device(config["training"]["device"])

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
            / f"{config['model'].get('name', 'run')}_{run_date}"
        )

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)
        self.logger.info(f"Data path set to: {self.data_path}")

        self.checkpoint_path = self.data_path / "model_checkpoints"
        self.checkpoint_at = config["training"].get("checkpoint_at", None)
        self.latest_checkpoint = None
        # training and evaluation functions
        self.validator = validator
        self.tester = tester
        self.model = None
        self.optimizer = None

        with open(self.data_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        self.logger.info("Trainer initialized")
        self.logger.debug(f"Configuration: {self.config}")

    def initialize_model(self) -> Any:
        """Initialize the model for training.

        Returns:
            Any: The initialized model.
        """
        if self.model is not None:
            return self.model
        # try:
        model = gnn_model.GNNModel.from_config(self.config["model"])
        model = model.to(self.device)
        self.model = model
        self.logger.info("Model initialized to device: {}".format(self.device))
        return self.model

    def initialize_optimizer(self) -> torch.optim.Optimizer | None:
        """Initialize the optimizer for training.

        Raises:
            RuntimeError: If the model is not initialized.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """

        if self.model is None:
            raise RuntimeError(
                "Model must be initialized before initializing optimizer."
            )

        if self.optimizer is not None:
            return self.optimizer

        try:
            lr = self.config["training"].get("learning_rate", 0.001)
            weight_decay = self.config["training"].get("weight_decay", 0.0001)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            self.optimizer = optimizer
            self.logger.info(
                f"Optimizer initialized with learning rate: {lr} and weight decay: {weight_decay}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing optimizer: {e}")
        return self.optimizer

    def prepare_dataloaders(
        self, dataset: Dataset, split: list[float] = [0.8, 0.1, 0.1]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare the data loaders for training, validation, and testing.

        Args:
            dataset (Dataset): The dataset to prepare.
            split (list[float], optional): The split ratios for training, validation, and test sets. Defaults to [0.8, 0.1, 0.1].

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The data loaders for training, validation, and testing.
        """
        train_size = int(len(dataset) * split[0])
        val_size = int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size

        if not np.isclose(np.sum(split), 1.0, rtol=1e-05, atol=1e-08, equal_nan=False):
            raise ValueError(f"Split ratios must sum to 1.0. Provided split: {split}")

        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        )

        train_loader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=self.config["training"].get("pin_memory", True),
            drop_last=self.config["training"].get("drop_last", False),
            prefetch_factor=self.config["training"].get("prefetch_factor", None),
            shuffle=self.config["training"].get("shuffle", True),
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

        #
        output_size = len(self.config["model"]["classifier"]["output_dims"])

        losses = torch.zeros(
            len(train_loader), output_size, dtype=torch.float32, device=self.device
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
            loss = self.criterion(outputs, data)

            self.logger.debug(f"    Backpropagating loss: {loss.item()}")
            loss.backward()

            optimizer.step()

            losses[i, :] = loss

        return losses

    def _check_model_status(self, eval_data: list[Any] | torch.Tensor) -> bool:
        """Check the status of the model during training.

        Args:
            eval_data (list[Any]): The evaluation data from the training epoch.

        Returns:
            bool: Whether the training should stop early.
        """
        if (
            self.checkpoint_at is not None
            and self.epoch % self.checkpoint_at == 0
            and self.epoch > 0
        ):
            self.save_checkpoint()

        if self.early_stopping is not None:
            if self.early_stopping(eval_data):
                self.logger.debug(f"Early stopping at epoch {self.epoch}.")
                self.save_checkpoint(name_addition="early_stopping")
                return True

            if self.early_stopping.found_better:
                self.logger.debug(f"Found better model at epoch {self.epoch}.")
                self.save_checkpoint(name_addition="current_best")
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

        self.model = self.initialize_model()

        optimizer = self.initialize_optimizer()

        if optimizer is None:
            raise AttributeError(
                "Error, optimizer must be successfully initialized before running training"
            )

        total_training_data = torch.zeros(num_epochs, 2, dtype=torch.float32)

        for epoch in range(0, num_epochs):
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")

            self.model.train()

            epoch_data = self._run_train_epoch(self.model, optimizer, train_loader)

            # collect mean and std for each epoch
            total_training_data[epoch, :] = torch.Tensor(
                [epoch_data.mean(dim=0).item(), epoch_data.std(dim=0).item()]
            )

            self.logger.info(
                f"  Completed epoch {self.epoch}. training loss: {total_training_data[self.epoch, 0]} +/- {total_training_data[self.epoch, 1]}."
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
        self,
        test_loader: DataLoader,
    ) -> Collection[Any]:
        """Run testing phase.

        Args:
            test_loader (DataLoader): The data loader for the test set.

        Raises:
            RuntimeError: If the model is not initialized.
            RuntimeError: If the test data is not available.

        Returns:
            Collection[Any]: A collection of test results that can be scalars, tensors, lists, dictionaries or any other data type that the tester might return.
        """
        self.logger.info("Starting testing process.")
        if self.model is None:
            raise RuntimeError("Model must be initialized before testing.")
        self.model.eval()
        if self.tester is None:
            raise RuntimeError("Tester must be initialized before testing.")
        test_result = self.tester.test(self.model, test_loader)
        self.tester.report(test_result)
        self.logger.info("Testing process completed.")
        return self.tester.data

    def save_checkpoint(self, name_addition: str = ""):
        """Save model checkpoint.

        Raises:
            ValueError: If the model is not initialized.
            ValueError: If the model configuration does not contain 'name'.
            ValueError: If the training configuration does not contain 'checkpoint_path'.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before saving checkpoint.")

        self.logger.info(
            f"Saving checkpoint for model {self.config['model'].get('name', ' model')} at epoch {self.epoch} to {self.checkpoint_path}"
        )
        outpath = (
            self.checkpoint_path
            / f"{self.config['model'].get('name', 'model')}_epoch_{self.epoch}_{name_addition}.pt"
        )

        if outpath.exists() is False:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory {outpath.parent} for checkpoint.")

        self.latest_checkpoint = outpath
        self.model.save(outpath)

    def load_checkpoint(self, epoch: int, name_addition: str = "") -> None:
        """Load model checkpoint to the device given

        Args:
            epoch (int): The epoch number to load.

        Raises:
            RuntimeError: If the model is not initialized.
        """

        if self.model is None:
            raise RuntimeError("Model must be initialized before loading checkpoint.")

        loadpath = (
            Path(self.checkpoint_path)
            / f"{self.config['model'].get('name', 'model')}_epoch_{epoch}_{name_addition}.pt"
        )

        if not loadpath.exists():
            raise FileNotFoundError(f"Checkpoint file {loadpath} does not exist.")

        self.model = gnn_model.GNNModel.load(loadpath)
