from typing import Callable, Any, Tuple
import torch
from torch.optim.optimizer import Optimizer as Optimizer
import torch.multiprocessing as mp
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from collections.abc import Collection
import os

from . import gnn_model
from .evaluate import DefaultValidator, DefaultTester


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


# this function behaves like a factory function, which is why it is capitalized
def DistributedDataLoader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DistributedSampler]:
    """Create a distributed data loader for training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        batch_size (int): The batch size to use.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory for the data loader. Defaults to True.
        rank (int, optional): The rank of the current process. Defaults to 0.
        world_size (int, optional): The total number of processes. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        seed (int, optional): The random seed for shuffling. Defaults to 42.

    Returns:
        DataLoader: The data loader for the distributed training.
    """

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )

    return dataloader, sampler


def DistributedModel(
    model: torch.nn.Module, rank: int, output_device: int | None = None, **ddp_kwargs
) -> torch.nn.Module:
    """Create a distributed data parallel model for training.

    Args:
        model (torch.nn.Module): The model to train.
        rank (int): The rank of the current process.
        output_device (int | None, optional): The device to output results to. Defaults to None.

    Returns:
        torch.nn.Module: The distributed data parallel model.
    """
    ddp_model = DDP(model, device_ids=[rank], output_device=output_device, **ddp_kwargs)
    return ddp_model


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
            all(x in config for x in ["training", "data", "model", "val", "test"])
            is False
        ):
            raise ValueError(
                "Configuration must contain 'training' and 'data' sections."
            )

        self.config = config
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
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.epoch = 0
        self.checkpoint_at = config["training"].get("checkpoint_at", None)

        # training and evaluation functions
        self.validator = validator
        self.tester = tester
        self.model = None
        self.optimizer = None

    def initialize_model(self) -> Any:
        """Initialize the model for training.

        Returns:
            Any: The initialized model.
        """
        model = gnn_model.GNNModel.from_config(self.config["model"])
        model = model.to(self.device)
        return model

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

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        return optimizer

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

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["training"]["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
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

        eval_data = []
        # training run
        for batch in train_loader:
            optimizer.zero_grad()
            data = batch.to(self.device, non_blocking=True)
            outputs = self._evaluate_batch(model, data)
            loss = self.criterion(outputs, data)
            loss.backward()
            optimizer.step()
            if isinstance(loss, torch.Tensor):
                eval_data.append(loss.item())
            else:
                eval_data.append(loss)
        return eval_data

    def _check_model_status(self, eval_data: list[Any]) -> bool:
        """Check the status of the model during training.

        Args:
            eval_data (list[Any]): The evaluation data from the training epoch.

        Returns:
            bool: Whether the training should stop early.
        """

        if self.checkpoint_at is not None and self.epoch % self.checkpoint_at == 0:
            self.save_checkpoint()

        if self.early_stopping is not None:
            if self.early_stopping(eval_data):
                print(f"Early stopping at epoch {self.epoch}.")
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

        # training loop
        num_epochs = self.config["training"].get("num_epochs", 100)
        self.model = self.initialize_model()
        optimizer = self.initialize_optimizer()
        total_training_data = []
        for _ in range(0, num_epochs):
            self.model.train()
            epoch_data = self._run_train_epoch(self.model, optimizer, train_loader)
            total_training_data.append(epoch_data)

            # evaluation run on validation set
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                self.validator.report(*validation_result)

            should_stop = self._check_model_status(
                self.validator.data if self.validator else total_training_data,
            )
            if should_stop:
                break

            self.epoch += 1

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
        if self.model is None:
            raise RuntimeError("Model must be initialized before testing.")
        self.model.eval()
        if self.tester is None:
            raise RuntimeError("Tester must be initialized before testing.")
        test_result = self.tester.test(self.model, test_loader)
        self.tester.report(*test_result)
        return test_result

    def save_checkpoint(self):
        """Save model checkpoint.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before saving checkpoint.")

        torch.save(
            self.model.state_dict(),
            self.config["training"].get(
                "checkpoint_path",
                f"{self.config['model']['name']}_epoch{self.epoch}.pt",
            ),
        )

    def load_checkpoint(self, epoch: int) -> None:
        """Load model checkpoint.

        Args:
            epoch (int): The epoch number to load.

        Raises:
            RuntimeError: If the model is not initialized.
        """

        if self.model is None:
            raise RuntimeError("Model must be initialized before saving checkpoint.")
        self.model.load_state_dict(
            torch.load(
                self.config["training"].get(
                    "checkpoint_path", f"checkpoint_{epoch}.pth"
                )
            )
        )


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
        super().__init__(
            config,
            criterion,
            apply_model,
            early_stopping,
            validator,
            tester,
        )

        if "parallel" not in config:
            raise ValueError("Configuration must contain 'parallel' section for DDP.")

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        self.rank = rank
        self.device = rank  # Set the device to the rank for DDP
        self.world_size = config["parallel"].get("world_size", 1)
        self.device = self.rank  # Set the device to the rank for DDP

    def initialize_model(self) -> Any:
        """Initialize the model for training.

        Returns:
            Any: The initialized model.
        """
        model = gnn_model.GNNModel.from_config(self.config["model"])
        model = model.to(self.rank)
        model = DDP(
            model,
            device_ids=[self.device],
            output_device=self.config["parallel"].get("output_device", None),
            find_unused_parameters=self.config["parallel"].get(
                "find_unused_parameters", False
            ),
        )
        return model

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
            batch_size=self.config["train"]["batch_size"],
            sampler=self.train_sampler,
            num_workers=self.config["train"].get("data_num_workers", 0),
            pin_memory=self.config["train"].get("pin_memory", True),
            drop_last=self.config["train"].get("drop_last", False),
            prefetch_factor=self.config["train"].get("prefetch_factor", 2),
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.config["val"]["batch_size"],
            num_workers=self.config["val"].get("data_num_workers", 0),
            pin_memory=self.config["val"].get("pin_memory", True),
            drop_last=self.config["val"].get("drop_last", False),
            prefetch_factor=self.config["val"].get("prefetch_factor", 2),
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.config["test"]["batch_size"],
            num_workers=self.config["test"].get("data_num_workers", 0),
            pin_memory=self.config["test"].get("pin_memory", True),
            drop_last=self.config["test"].get("drop_last", False),
            prefetch_factor=self.config["test"].get("prefetch_factor", 2),
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
        initialize_ddp(
            self.rank,
            self.config["parallel"]["world_size"],
            master_addr=self.config["parallel"].get("master_addr", "localhost"),
            master_port=self.config["parallel"].get("master_port", "12345"),
            backend=self.config["parallel"].get("backend", "nccl"),
        )

        self.model = self.initialize_model()
        self.optimizer = self.initialize_optimizer()

        num_epochs = self.config["training"].get("num_epochs", 100)

        total_training_data = []
        all_training_data = [None for _ in range(self.world_size)]
        all_validation_data = [None for _ in range(self.world_size)]
        for _ in range(0, num_epochs):
            self.model.train()
            train_loader.sampler.set_epoch(self.epoch)
            self._run_train_epoch(self.model, self.optimizer, train_loader)

            # evaluation run on validation set
            self.model.eval()
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                if self.rank == 0:
                    self.validator.report(*validation_result)

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
        cleanup_ddp()
        return all_training_data, all_validation_data


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

    def run_training_ddp(
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
            if rank == 0:
                test_data = trainer.run_test(test_loader)
                result_queue.put(
                    {
                        "training_data": train_data,
                        "validation_data": val_data,
                        "test_data": test_data,
                    }
                )
        except Exception as e:
            print(f"Rank {rank} crashed: {e}", flush=True)
            result_queue.put(None)
        finally:
            cleanup_ddp()

    # We have a function to run in parallel now for each process.
    # We send one for each device in our 'world' to a process to run now
    mp.spawn(
        run_training_ddp,
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
