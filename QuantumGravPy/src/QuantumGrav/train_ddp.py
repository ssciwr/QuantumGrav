from typing import Any, Tuple
from collections.abc import Collection
import pandas as pd
import os
import jsonschema
import logging
import yaml
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from . import evaluate
from . import early_stopping
from . import gnn_model
from . import train

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
    """Distributed Data Parallel (DDP) Trainer for training GNN models across multiple processes."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Model trainer class Configuration",
        "type": "object",
        "definitions": train.Trainer.schema.get("definitions", {}),
        "properties": {
            "parallel": {
                "type": "object",
                "properties": {
                    "world_size": {"type": "integer", "minimum": 1},
                    "output_device": {"type": ["integer", "null"]},
                    "find_unused_parameters": {"type": "boolean"},
                    "rank": {"type": "integer", "minimum": 0},
                    "master_addr": {"type": "string"},
                    "master_port": {"type": "string"},
                },
                "required": ["world_size"],
            },
            "log_level": train.Trainer.schema["properties"]["log_level"],
            "training": train.Trainer.schema["properties"]["training"],
            "model": train.Trainer.schema["properties"]["model"],
            "validation": train.Trainer.schema["properties"]["validation"],
            "testing": train.Trainer.schema["properties"]["testing"],
            "early_stopping": train.Trainer.schema["properties"]["early_stopping"],
            "apply_model": train.Trainer.schema["properties"]["apply_model"],
            "criterion": train.Trainer.schema["properties"]["criterion"],
        },
        "required": [
            "parallel",
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
            validator (Validator | None, optional): The validator for model evaluation. Defaults to None.
            tester (Tester | None, optional): The tester for model testing. Defaults to None.

        Raises:
            ValueError: If the configuration is invalid.
        """
        jsonschema.validate(instance=config, schema=self.schema)
        logger = logging.getLogger(__name__)
        logger.setLevel(config.get("log_level", logging.INFO))
        logger.info("Initializing TrainerDDP instance")

        criterion = config["criterion"]
        apply_model = config.get("apply_model")
        seed = config["training"]["seed"]

        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rank)

        if torch.cuda.is_available() and config["training"]["device"] != "cpu":
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_path = (
            Path(config["training"]["path"]) / f"{config.get('name', 'run')}_{run_date}"
        )
        if not data_path.exists():
            data_path.mkdir(parents=True)

        if "early_stopping" in config:
            try:
                early_stopper = early_stopping.DefaultEarlyStopping.from_config(
                    config["early_stopping"]
                )
            except Exception:
                early_stopping_kwargs = deepcopy(config["early_stopping"]["kwargs"])
                early_stopper = config["early_stopping"]["type"](
                    *config["early_stopping"]["args"],
                    **early_stopping_kwargs,
                )
        else:
            early_stopper = None

        if "validation" in config:
            try:
                validator = evaluate.Validator.from_config(
                    config["validation"]["validator"]
                )
            except Exception:
                validator = config["validation"]["validator"]["type"](
                    *config["validation"]["validator"]["args"],
                    **config["validation"]["validator"]["kwargs"],
                )
        else:
            validator = None

        if "testing" in config:
            try:
                tester = evaluate.Tester.from_config(config["testing"]["tester"])
            except Exception:
                tester = config["testing"]["tester"]["type"](
                    *config["testing"]["tester"]["args"],
                    **config["testing"]["tester"]["kwargs"],
                )
        else:
            tester = None

        with open(data_path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        self.checkpoint_at = config["training"].get("checkpoint_at", None)
        self.latest_checkpoint = None
        self.sampler = None
        self.config = config
        self.logger = logger
        self.criterion = criterion
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.apply_model = apply_model
        self.validator = validator
        self.early_stopper = early_stopper
        self.tester = tester
        self.seed = seed
        self.device = device
        self.epoch = 0
        self.data_path = data_path
        self.checkpoint_path = data_path / "checkpoints"

        # initialize the systems differently on each process/rank
        self.rank = rank
        self.world_size = config["parallel"]["world_size"]
        self.logger.info("Initialized DDP trainer")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainerDDP":
        """Create a DDP trainer from configuration.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Returns:
            TrainerDDP: Configured trainer instance.
        """
        rank = config.get("parallel", {}).get("rank", 0)
        return cls(rank=rank, config=config)

    def initialize_model(self) -> DDP:
        """Initialize the model for training.

        Returns:
            DDP: The initialized model.
        """
        model = gnn_model.GNNModel.from_config(self.config["model"])

        if self.device.type == "cpu" or (
            isinstance(self.device, torch.device) and self.device.type == "cpu"
        ):
            # For CPU training, device_ids should be None
            d_id = None
            o_id = None
        else:
            # For CUDA, device_ids should be a list of device indices
            # Extract device index from the device string (e.g., "cuda:0" -> 0)
            device_idx = (
                int(str(self.device).split(":")[1])
                if ":" in str(self.device)
                else self.rank
            )
            d_id = [device_idx]
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

    def _check_model_status(self, eval_data: pd.DataFrame | list[torch.Tensor]) -> bool:
        """Check the status of the model during evaluation.

        Args:
            eval_data (pd.DataFrame): The evaluation data to check.

        Returns:
            bool: Whether the model training should stop.
        """
        should_stop = False
        if self.rank == 0:
            should_stop = super()._check_model_status(eval_data)
        return should_stop

    def save_checkpoint(self, name_addition: str = ""):
        """Save model checkpoint on rank 0 only.

        Args:
            name_addition (str): Optional checkpoint suffix.
        """
        if self.rank == 0:
            super().save_checkpoint(name_addition=name_addition)

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
            if train_loader.sampler:
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

            # Ensure all processes have completed the epoch before checking status
            dist.barrier()

            should_stop = False
            if self.rank == 0:
                should_stop = self._check_model_status(
                    self.validator.data if self.validator else total_training_data,
                )

            # Broadcast early stopping decision from rank 0 to all ranks
            object_list = [should_stop]
            dist.broadcast_object_list(object_list, src=0, device=self.device)
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
