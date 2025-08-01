from typing import Callable, Any, Tuple
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from collections.abc import Collection
import os
import numpy as np
from pathlib import Path
import logging
from . import gnn_model
from .evaluate import DefaultValidator, DefaultTester
import tqdm
import yaml
from datetime import datetime


def initialize_ddp(
    rank: int,
    worldsize: int,
    master_addr: str = "localhost",
    master_port: str = "12345",
    backend: str = "nccl",
) -> None:
    """Initialize the distributed process group. This assumes one process per GPU.

    Args:
        rank (int): The rank of the current process.
        worldsize (int): The total number of processes.
        master_addr (str, optional): The address of the master process. Defaults to "localhost". This needs to be the ip of the master node if you are running on a cluster.
        master_port (str, optional): The port of the master process. Defaults to "12345". Choose a high port if you are running multiple jobs on the same machine to avoid conflicts. If running on a cluster, this should be the port that the master node is listening on.
        backend (str, optional): The backend to use for distributed training. Defaults to "nccl".

    Raises:
        RuntimeError: If the environment variables MASTER_ADDR and MASTER_PORT are already set.
    """
    if "MASTER_ADDR" in os.environ or "MASTER_PORT" in os.environ:
        raise RuntimeError(
            "Environment variables MASTER_ADDR and MASTER_PORT are already set. Please unset them before initializing."
        )
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=worldsize)


def cleanup_ddp() -> None:
    """Clean up the distributed process group."""
    dist.destroy_process_group()
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        # training and evaluation functions
        criterion: Callable,
        apply_model: Callable | None = None,
        # training evaluation and reporting
        early_stopping: Callable[[list[dict[str, Any]]], bool] | None = None,
        validator: DefaultValidator | None = None,
        tester: DefaultTester | None = None,
    ):
        """Initialize the trainer.

        Args:
            config (dict[str, Any]): The configuration dictionary.
            criterion (Callable): The loss function to use.
            apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
            early_stopping (Callable[[list[dict[str, Any]]], bool] | None, optional): A function for early stopping. Defaults to None.
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
                "Configuration must contain 'training', 'model', 'validatino' and 'testing' sections."
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

        # early stopping parameters
        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.early_stopping_counter = 0

        # parameters for finding out which model is best
        self.best_score = None
        self.best_epoch = 0
        self.epoch = 0

        # date and time of run:
        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.data_path = Path(self.config["training"]["path"]) / run_date

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
        # except Exception as e:
        #     self.logger.error(f"Error initializing model: {e}")
        #     return None

    def initialize_optimizer(self) -> torch.optim.Optimizer:
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
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=self.config["training"].get("pin_memory", True),
            drop_last=self.config["training"].get("drop_last", False),
            prefetch_factor=self.config["training"].get("prefetch_factor", None),
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["validation"]["batch_size"],
            num_workers=self.config["validation"].get("num_workers", 0),
            pin_memory=self.config["validation"].get("pin_memory", True),
            drop_last=self.config["validation"].get("drop_last", False),
            prefetch_factor=self.config["validation"].get("prefetch_factor", None),
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["testing"]["batch_size"],
            num_workers=self.config["testing"].get("num_workers", 0),
            pin_memory=self.config["testing"].get("pin_memory", True),
            drop_last=self.config["testing"].get("drop_last", False),
            prefetch_factor=self.config["testing"].get("prefetch_factor", None),
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
    ) -> list[Any]:
        """Run a single training epoch.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
            train_loader (DataLoader): The data loader for the training set.
        Raises:
            RuntimeError: If the model is not initialized.
            RuntimeError: If the optimizer is not initialized.

        Returns:
            list[Any]: The training loss for each batch.
        """

        if model is None:
            raise RuntimeError("Model must be initialized before training.")

        if optimizer is None:
            raise RuntimeError("Optimizer must be initialized before training.")

        output_size = len(self.config["model"]["classifier"]["output_dims"])
        losses = torch.zeros(
            len(train_loader), output_size, dtype=torch.float32, device=self.device
        )
        # training run
        for i, batch in enumerate(
            tqdm.tqdm(train_loader, desc=f"Training Epoch {self.epoch}")
        ):
            optimizer.zero_grad()
            self.logger.debug(f"  Moving batch to device: {self.device}")
            data = batch.to(self.device)
            outputs = self._evaluate_batch(model, data)
            self.logger.debug("  Computing loss")
            loss = self.criterion(outputs, data)
            self.logger.debug(f"  Backpropagating loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            losses[i, :] = loss
        return losses

    def _check_model_status(self, eval_data: list[Any]) -> bool:
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
                self.logger.info(f"Early stopping at epoch {self.epoch}.")
                self.save_checkpoint()
                return True

        return False

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[Collection[Any], Collection[Any]]:
        """Run the training process.

        Args:
            train_loader (DataLoader): The data loader for the training set.
            val_loader (DataLoader): The data loader for the validation set.

        Returns:
            Tuple[Collection[Any], Collection[Any]]: The training and validation results.
        """
        self.logger.info("Starting training process.")
        # training loop
        num_epochs = self.config["training"]["num_epochs"]

        self.model = self.initialize_model()
        optimizer = self.initialize_optimizer()
        total_training_data = torch.zeros(num_epochs, 2, dtype=torch.float32)
        for epoch in range(0, num_epochs):
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")
            self.model.train()
            epoch_data = self._run_train_epoch(self.model, optimizer, train_loader)

            # collect mean and std for each epoch
            total_training_data[epoch, :] = torch.Tensor(
                [epoch_data.mean(dim=0), epoch_data.std(dim=0)]
            )

            self.logger.info(
                f"  Completed epoch {self.epoch}. training loss: {total_training_data[self.epoch, 0]} +/- {total_training_data[self.epoch, 1]}."
            )

            # evaluation run on validation set
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                self.validator.report(validation_result)

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
        torch.save(self.model.state_dict(), outpath)

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
        return test_result

    def save_checkpoint(self):
        """Save model checkpoint.

        Raises:
            ValueError: If the model is not initialized.
            ValueError: If the model configuration does not contain 'name'.
            ValueError: If the training configuration does not contain 'checkpoint_path'.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before saving checkpoint.")

        if "name" not in self.config["model"]:
            raise ValueError(
                "Model configuration must contain 'name' to save checkpoint."
            )

        self.logger.info(
            f"Saving checkpoint for model {self.config['model']['name']} at epoch {self.epoch} to {self.checkpoint_path}"
        )
        outpath = (
            self.checkpoint_path
            / f"{self.config['model']['name']}_epoch_{self.epoch}.pt"
        )

        if outpath.exists() is False:
            outpath.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory {outpath.parent} for checkpoint.")

        self.latest_checkpoint = outpath
        torch.save(self.model.state_dict(), outpath)

    def load_checkpoint(self, epoch: int) -> None:
        """Load model checkpoint to the device given

        Args:
            epoch (int): The epoch number to load.

        Raises:
            RuntimeError: If the model is not initialized.
        """

        if self.model is None:
            raise RuntimeError("Model must be initialized before saving checkpoint.")

        loadpath = (
            Path(self.config["training"]["checkpoint_path"])
            / f"{self.config['model']['name']}_epoch_{epoch}.pt"
        )

        if not loadpath.exists():
            raise FileNotFoundError(f"Checkpoint file {loadpath} does not exist.")

        self.model.load_state_dict(torch.load(loadpath, map_location=self.device))


class TrainerDDP(Trainer):
    def __init__(
        self,
        rank: int,
        config: dict[str, Any],
        # training and evaluation functions
        criterion: Callable,
        apply_model: Callable | None = None,
        # training evaluation and reporting
        early_stopping: Callable[[list[dict[str, Any]]], bool] | None = None,
        validator: DefaultValidator | None = None,
        tester: DefaultTester | None = None,
    ):
        """Initialize the distributed data parallel (DDP) trainer.

        Args:
            rank (int): The rank of the current process.
            config (dict[str, Any]): The configuration dictionary.
            criterion (Callable): The loss function.
            apply_model (Callable | None, optional): The function to apply the model. Defaults to None.
            early_stopping (Callable[[list[dict[str, Any]]], bool] | None, optional): The early stopping function. Defaults to None.
            validator (DefaultValidator | None, optional): The validator for model evaluation. Defaults to None.
            tester (DefaultTester | None, optional): The tester for model testing. Defaults to None.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if "parallel" not in config:
            raise ValueError("Configuration must contain 'parallel' section for DDP.")

        super().__init__(
            config,
            criterion,
            apply_model,
            early_stopping,
            validator,
            tester,
        )
        # initialize the systems differently on each process/rank
        torch.manual_seed(self.seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + rank)

        if torch.cuda.is_available() and config["training"]["device"] != "cpu":
            torch.cuda.set_device(rank)
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        self.rank = rank
        self.world_size = config["parallel"]["world_size"]
        self.logger.info("Initialized DDP trainer")

    def initialize_model(self) -> DDP:
        """Initialize the model for training.

        Returns:
            DDP: The initialized model.
        """
        model = gnn_model.GNNModel.from_config(self.config["model"])

        if self.device == "cpu" or (
            isinstance(self.device, torch.device) and self.device.type == "cpu"
        ):
            d_id = None
            o_id = None
        else:
            d_id = [
                self.device,
            ]
            o_id = self.config["parallel"].get("output_device", None)
        model = DDP(
            model,
            device_ids=d_id,
            output_device=o_id,
            find_unused_parameters=self.config["parallel"].get(
                "find_unused_parameters", False
            ),
        )
        self.model = model.to(self.device, non_blocking=True)
        self.logger.info(f"Model initialized on device: {self.device}")
        return self.model

    def prepare_dataloaders(
        self, dataset: Dataset, split: list[float] = [0.8, 0.1, 0.1]
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
    ]:
        """Prepare the data loaders for training, validation, and testing.

        Args:
            dataset (Dataset): The dataset to split.
            split (list[float], optional): The proportions for train/val/test split. Defaults to [0.8, 0.1, 0.1].

        Returns:
            Tuple[ DataLoader, DataLoader, DataLoader, ]: The data loaders for training, validation, and testing.
        """
        train_size = int(len(dataset) * split[0])
        val_size = int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size
        self.logger.info(
            f"Preparing data loaders with split: {split}, train size: {train_size}, val size: {val_size}, test size: {test_size}"
        )
        if (
            np.isclose(np.sum(split), 1.0, rtol=1e-05, atol=1e-08, equal_nan=False)
            is False
        ):
            raise ValueError(
                "Split ratios must sum to 1.0. Provided split: {}".format(split)
            )

        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        )

        # samplers are needed to distribute the data across processes in such a way that each process gets a unique subset of the data
        self.train_sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        self.val_sampler = torch.utils.data.DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.test_sampler = torch.utils.data.DistributedSampler(
            self.test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        # make the data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            sampler=self.train_sampler,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=self.config["training"].get("pin_memory", True),
            drop_last=self.config["training"].get("drop_last", False),
            prefetch_factor=self.config["training"].get("prefetch_factor", None),
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.config["validation"]["batch_size"],
            num_workers=self.config["validation"].get("num_workers", 0),
            pin_memory=self.config["validation"].get("pin_memory", True),
            drop_last=self.config["validation"].get("drop_last", False),
            prefetch_factor=self.config["validation"].get("prefetch_factor", None),
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.config["testing"]["batch_size"],
            num_workers=self.config["testing"].get("num_workers", 0),
            pin_memory=self.config["testing"].get("pin_memory", True),
            drop_last=self.config["testing"].get("drop_last", False),
            prefetch_factor=self.config["testing"].get("prefetch_factor", None),
        )
        self.logger.info(
            f"Data loaders prepared with splits: {split} and dataset sizes: {len(self.train_dataset)}, {len(self.val_dataset)}, {len(self.test_dataset)}"
        )
        return train_loader, val_loader, test_loader

    def _check_model_status(self, eval_data: list[Any]) -> bool:
        """Check the status of the model during evaluation.

        Args:
            eval_data (list[Any]): The evaluation data to check.

        Returns:
            bool: Whether the model training should stop.
        """
        should_stop = False
        if self.rank == 0:
            should_stop = super()._check_model_status(eval_data)
        return should_stop

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[list[Any], list[Any]]:
        """
        Run the training loop for the distributed model. This will synchronize for validation. No testing is performed in this function. The model will only be checkpointed and early stopped on the 'rank' 0 process.

        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.

        Returns:
            Tuple[list[Any], list[Any]]: The training and validation results.
        """

        self.model = self.initialize_model()
        self.optimizer = self.initialize_optimizer()

        num_epochs = self.config["training"]["num_epochs"]
        self.logger.info("Starting training process.")

        total_training_data = []
        all_training_data = [None for _ in range(self.world_size)]
        all_validation_data = [None for _ in range(self.world_size)]
        for _ in range(0, num_epochs):
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")
            self.model.train()
            train_loader.sampler.set_epoch(self.epoch)
            epoch_data = self._run_train_epoch(self.model, self.optimizer, train_loader)
            total_training_data.append(epoch_data)  # TODO: check if this works in DDP

            # evaluation run on validation set
            self.model.eval()
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                if self.rank == 0:
                    self.validator.report(validation_result)

            dist.barrier()  # Ensure all processes have completed the epoch before checking status
            should_stop = self._check_model_status(
                self.validator.data if self.validator else total_training_data,
            )
            if should_stop:
                break

            self.epoch += 1

        dist.barrier()
        dist.all_gather_object(all_training_data, total_training_data)
        dist.all_gather_object(
            all_validation_data, self.validator.data if self.validator else []
        )
        self.logger.info("Training process completed.")
        return all_training_data, all_validation_data


def __run_training_loop_ddp__(
    rank,
    config,
    dataset,
    split,
    result_queue: mp.Queue,
    # training and evaluation functions
    criterion: Callable,
    apply_model: Callable | None = None,
    # training evaluation and reporting
    early_stopping: Callable[[list[dict[str, Any]]], bool] | None = None,
    validator: DefaultValidator | None = None,
    tester: DefaultTester | None = None,
):
    """Train a single model on a single process.

    Args:
        rank (int): The rank of the process.
        config (dict[str, Any]): Configuration dictionary.
        dataset (Dataset): The dataset to train on.
        split (list[float]): The train/validation/test split ratios.
        result_queue (mp.Queue): Queue to collect results.
        criterion (Callable): The loss function.
        apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
        early_stopping (Callable[[list[dict[str, Any]]], bool] | None, optional): Early stopping criteria. Defaults to None.
        validator (DefaultValidator | None, optional): Validator class instance for model evaluation. Defaults to None.
        tester (DefaultTester | None, optional): Tester class instance for model evaluation. Defaults to None.
    """
    try:
        initialize_ddp(
            rank,
            config["parallel"]["world_size"],
            master_addr=config["parallel"].get("master_addr", "localhost"),
            master_port=config["parallel"].get("master_port", "12345"),
            backend=config["parallel"].get("backend", "nccl"),
        )

        trainer = TrainerDDP(
            rank,
            config,
            criterion,
            apply_model=apply_model,
            early_stopping=early_stopping,
            validator=validator,
            tester=tester,
        )
        train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
            dataset, split=split
        )

        train_data, val_data = trainer.run_training(
            train_loader,
            val_loader,
        )
        test_data = None
        test_data = trainer.run_test(test_loader)

        result_queue.put(
            {
                "training_data": train_data,
                "validation_data": val_data,
                "test_data": test_data,
            }
        )

        cleanup_ddp()
    except Exception as e:
        print(f"Rank {rank} crashed: {e}", flush=True)
        result_queue.put(None)
        if dist.is_initialized():
            cleanup_ddp()


def train_parallel(
    config: dict[str, Any],
    dataset: Dataset,
    split: list[float],
    # training and evaluation functions
    criterion: Callable,
    apply_model: Callable | None = None,
    # training evaluation and reporting
    early_stopping: Callable[[list[dict[str, Any]]], bool] | None = None,
    validator: DefaultValidator | None = None,
    tester: DefaultTester | None = None,
) -> Tuple[list[Any], list[Any], list[Any]]:
    """Train a model in a parallel fashion using DDP.

    Args:
        config (dict[str, Any]): Configuration dictionary.
        dataset (Dataset): The dataset to train on.
        split (list[float]): The train/validation/test split ratios.
        criterion (Callable): The loss function.
        apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
        early_stopping (Callable[[list[dict[str, Any]]], bool] | None, optional): Early stopping criteria. Defaults to None.
        validator (DefaultValidator | None, optional): Validator class instance for model evaluation. Defaults to None.
        tester (DefaultTester | None, optional): Tester class instance for model evaluation. Defaults to None.

    Returns:
        Tuple[list[Any], list[Any], list[Any]]: Training, validation, and test results.
    """

    manager = mp.Manager()
    result_queue = manager.Queue()

    # We have a function to run in parallel now for each process.
    # We send one for each device in our 'world' to a process to run now
    mp.spawn(
        __run_training_loop_ddp__,
        args=(
            config,
            dataset,
            split,
            result_queue,
            criterion,
            apply_model,
            early_stopping,
            validator,
            tester,
        ),
        nprocs=config["parallel"]["world_size"],
        join=True,
    )

    # Get results from queue
    if not result_queue.empty():
        results = result_queue.get()
        return (
            results["training_data"],
            results["validation_data"],
            results["test_data"],
        )
    else:
        return [], [], []
