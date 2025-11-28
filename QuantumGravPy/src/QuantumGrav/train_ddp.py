from typing import Any, Tuple
from collections.abc import Collection

import numpy as np
import os
import copy

from . import gnn_model
from . import train

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import optuna


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
    if dist.is_initialized():
        raise RuntimeError("The distributed process group is already initialized.")
    else:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        dist.init_process_group(backend=backend, rank=rank, world_size=worldsize)


def cleanup_ddp() -> None:
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)


class TrainerDDP(train.Trainer):
    def __init__(
        self,
        rank: int,
        config: dict[str, Any],
        # training and evaluation functions
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

        trainer_config = copy.deepcopy(config)
        del trainer_config["parallel"]

        super().__init__(
            trainer_config,
        )

        self.config = config  # keep the full config including parallel section

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

        if self.device.type == "cpu" or (
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

    def _check_model_status(self, eval_data: list[Any] | torch.Tensor) -> bool:
        """Check the status of the model during evaluation.

        Args:
            eval_data (list[Any] | torch.Tensor): The evaluation data to check.

        Returns:
            bool: Whether the model training should stop.
        """
        should_stop = False
        if self.rank == 0:
            should_stop = super()._check_model_status(eval_data)
        return should_stop

    def save_checkpoint(self, name_addition: str = ""):
        """Save model checkpoint.

        Raises:
            ValueError: If the model is not initialized.
            ValueError: If the model configuration does not contain 'name'.
            ValueError: If the training configuration does not contain 'checkpoint_path'.
        """
        # TODO: check if this works really - it should save the best model that is there
        if self.rank == 0:
            if self.model is None:
                raise ValueError("Model must be initialized before saving checkpoint.")

            self.logger.info(
                f"Saving checkpoint for model model at epoch {self.epoch} to {self.checkpoint_path}"
            )
            outpath = self.checkpoint_path / f"model_{name_addition}.pt"

            if outpath.exists() is False:
                outpath.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory {outpath.parent} for checkpoint.")

            self.latest_checkpoint = outpath
            torch.save(self.model, outpath)

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: optuna.trial.Trial | None = None,
    ) -> Tuple[torch.Tensor | Collection[Any], torch.Tensor | Collection[Any]]:
        """
        Run the training loop for the distributed model. This will synchronize for validation. No testing is performed in this function. The model will only be checkpointed and early stopped on the 'rank' 0 process.

        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            trial (optuna.trial.Trial | None, optional): An Optuna trial for hyperparameter optimization.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor | Collection[Any], torch.Tensor | Collection[Any]]: The training and validation results.
        """

        self.model = self.initialize_model()
        self.optimizer = self.initialize_optimizer()

        num_epochs = self.config["training"]["num_epochs"]
        self.logger.info("Starting training process.")

        total_training_data = []
        all_training_data: list[Any] = [None for _ in range(self.world_size)]
        all_validation_data: list[Any] = [None for _ in range(self.world_size)]
        for _ in range(0, num_epochs):
            self.logger.info(f"  Current epoch: {self.epoch}/{num_epochs}")
            self.model.train()
            train_loader.sampler.set_epoch(self.epoch)
            epoch_data = self._run_train_epoch(self.model, self.optimizer, train_loader)
            total_training_data.append(epoch_data)

            # evaluation run on validation set
            self.model.eval()
            if self.validator is not None:
                validation_result = self.validator.validate(self.model, val_loader)
                if self.rank == 0:
                    self.validator.report(validation_result)

                    # integrate Optuna here for hyperparameter tuning
                    if trial is not None:
                        avg_sigma_loss = self.validator.data[self.epoch]
                        avg_loss = avg_sigma_loss[0]
                        trial.report(avg_loss, self.epoch)

                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

            dist.barrier()  # Ensure all processes have completed the epoch before checking status
            should_stop = self._check_model_status(
                self.validator.data if self.validator else total_training_data,
            )

            object_list = [should_stop]

            should_stop = dist.broadcast_object_list(
                object_list, src=0, device=self.device
            )
            should_stop = object_list[0]

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
