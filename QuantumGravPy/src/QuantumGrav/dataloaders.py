"""Helpers for creating datasets and dataloaders from config."""

from math import ceil, floor
from typing import Any
import logging

import jsonschema
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from . import base
from . import dataset_ondisk


DATA_CONFIG_SCHEMA = {
    "type": "object",
    "description": "Dataset configuration",
    "properties": {
        "pre_transform": {
            "description": "Callable applied before storing processed data",
        },
        "transform": {
            "description": "Callable applied when reading processed data",
        },
        "pre_filter": {
            "description": "Callable used to filter processed data",
        },
        "reader": {
            "description": "Callable that reads raw data from file",
        },
        "files": {
            "type": "array",
            "description": "List of zarr stores to read from",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "output": {
            "type": "string",
            "description": "Path to store preprocessed data",
        },
        "validate_data": {
            "type": "boolean",
            "description": "Whether to validate transformed data objects",
        },
        "n_processes": {
            "type": "integer",
            "minimum": 0,
            "description": "Processes to use for preprocessing",
        },
        "chunksize": {
            "type": "integer",
            "description": "Number of datapoints to process at once",
        },
        "shuffle": {
            "type": "boolean",
            "description": "Whether to shuffle the dataset",
        },
        "subset": {
            "type": "number",
            "description": "Fraction of the dataset to use",
        },
        "split": {
            "type": "array",
            "description": "Split ratios of the dataset",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 3,
        },
    },
    "required": ["output", "files", "reader"],
    "additionalProperties": False,
}


class DataLoaderFactory(base.Configurable):
    """Build datasets and dataloaders from training configuration."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DataLoaderFactory Configuration",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the training run",
            },
            "log_level": {
                "description": "Optional logging level (int or string, e.g. INFO)",
                "anyOf": [{"type": "integer"}, {"type": "string"}],
            },
            "data": DATA_CONFIG_SCHEMA,
            "training": {
                "type": "object",
                "description": "Training dataloader configuration",
                "properties": {
                    "data": DATA_CONFIG_SCHEMA,
                    "seed": {"type": "integer", "description": "Random seed"},
                    "batch_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Training DataLoader batch size",
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
                    "persistent_workers": {
                        "type": "boolean",
                        "description": "Keep workers alive between iterations",
                    },
                    "shuffle": {
                        "type": "boolean",
                        "description": "Shuffle training dataset",
                    },
                },
                "required": ["seed", "batch_size"],
                "additionalProperties": True,
            },
            "validation": {
                "type": "object",
                "description": "Validation dataloader configuration",
                "properties": {
                    "data": DATA_CONFIG_SCHEMA,
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
                },
                "required": ["batch_size"],
                "additionalProperties": True,
            },
            "testing": {
                "type": "object",
                "description": "Testing dataloader configuration",
                "properties": {
                    "data": DATA_CONFIG_SCHEMA,
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
                },
                "required": ["batch_size"],
                "additionalProperties": True,
            },
        },
        "required": ["training", "validation", "testing"],
        "additionalProperties": True,
    }

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the data loader factory.

        Args:
            config (dict[str, Any]): Configuration used for dataset and loader setup.
            logger (logging.Logger | None): Optional logger to reuse.
        """
        jsonschema.validate(instance=config, schema=type(self).schema)

        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(config.get("log_level", logging.INFO))
        self.nprng = np.random.default_rng(config["training"]["seed"])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DataLoaderFactory":
        """Create a data loader factory from configuration.

        Args:
            config (dict[str, Any]): Configuration used for dataset and loader setup.

        Returns:
            DataLoaderFactory: Configured factory instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(config.get("log_level", logging.INFO))
        return cls(config=config, logger=logger)

    def _build_dataset_from_config(
        self, data_config: dict[str, Any] | None, stage_name: str
    ) -> Dataset:
        """Build the configured on-disk dataset.

        Args:
            data_config (dict[str, Any] | None): Dataset config node to build from.
            stage_name (str): Name of the config node used for error reporting.

        Returns:
            Dataset: Dataset built from the provided config node.

        Raises:
            ValueError: If the config does not include the requested data section.
        """
        if data_config is None:
            raise ValueError(
                f"A '{stage_name}' data config section is required when no dataset is provided."
            )

        cfg = data_config
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
                self.nprng.integers(0, len(dataset), size=num_points).tolist()
            )

        # Keep the current dataset shuffle behavior unchanged.
        if cfg.get("shuffle"):
            dataset.shuffle()

        return dataset

    def _resolve_data_mode(self) -> str:
        """Resolve which supported config shape should build the datasets.

        Returns:
            str: Supported dataset resolution mode.

        Raises:
            ValueError: If the config mixes unsupported data nodes.
        """
        top_level_data = self.config.get("data")
        train_data = self.config["training"].get("data")
        validation_data = self.config["validation"].get("data")
        testing_data = self.config["testing"].get("data")

        if (
            train_data is not None
            and validation_data is not None
            and testing_data is not None
        ):
            return "stage_local"

        if (
            top_level_data is not None
            and train_data is None
            and validation_data is None
            and testing_data is not None
        ):
            return "shared_train_validation"

        if (
            top_level_data is not None
            and train_data is None
            and validation_data is None
            and testing_data is None
        ):
            return "top_level_full"

        if top_level_data is None and all(
            stage_data is None
            for stage_data in (train_data, validation_data, testing_data)
        ):
            raise ValueError(
                "A 'data' config section is required when no dataset is provided."
            )

        raise ValueError(
            "Unsupported data config. Use top-level 'data', top-level 'data' with "
            "'testing.data', or 'training.data', 'validation.data', and 'testing.data'."
        )

    def _validate_split(self, split: list[float], expected_parts: int) -> None:
        """Validate dataset split ratios.

        Args:
            split (list[float]): Split ratios to validate.
            expected_parts (int): Required number of split entries.

        Raises:
            ValueError: If the split has the wrong length or does not sum to 1.0.
        """
        if len(split) != expected_parts:
            raise ValueError(
                f"Split ratios must contain {expected_parts} values. Provided split: {split}"
            )

        if not np.isclose(np.sum(split), 1.0, rtol=1e-05, atol=1e-08, equal_nan=False):
            raise ValueError(f"Split ratios must sum to 1.0. Provided split: {split}")

    def _split_dataset(
        self, dataset: Dataset, split: list[float]
    ) -> tuple[torch.utils.data.Subset, ...]:
        """Split a dataset according to the configured resolution mode.

        Args:
            dataset (Dataset): Dataset to split.
            split (list[float]): Split ratios to apply.

        Returns:
            tuple[torch.utils.data.Subset, ...]: Split subsets.

        Raises:
            ValueError: If any resulting subset would be empty.
        """
        if len(split) == 3:
            # Keep the existing rounding behavior so split sizes stay stable.
            train_size = ceil(len(dataset) * split[0])
            val_size = floor(len(dataset) * split[1])
            test_size = len(dataset) - train_size - val_size

            if train_size == 0:
                raise ValueError("train size cannot be 0")

            if val_size == 0:
                raise ValueError("validation size cannot be 0")

            if test_size == 0:
                raise ValueError("test size cannot be 0")

            return torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

        train_size = ceil(len(dataset) * split[0])
        val_size = len(dataset) - train_size

        if train_size == 0:
            raise ValueError("train size cannot be 0")

        if val_size == 0:
            raise ValueError("validation size cannot be 0")

        return torch.utils.data.random_split(dataset, [train_size, val_size])

    def _prepare_datasets_from_config(
        self, split: list[float]
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Create datasets from the supported config modes.

        Args:
            split (list[float]): Fallback split ratios when the config omits them.

        Returns:
            tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.
        """
        mode = self._resolve_data_mode()

        if mode == "stage_local":
            return (
                self._build_dataset_from_config(
                    self.config["training"].get("data"), "training"
                ),
                self._build_dataset_from_config(
                    self.config["validation"].get("data"), "validation"
                ),
                self._build_dataset_from_config(
                    self.config["testing"].get("data"), "testing"
                ),
            )

        if mode == "shared_train_validation":
            data_config = self.config["data"]
            split = data_config.get("split", split)
            self._validate_split(split, expected_parts=2)

            # Split once so train and validation stay disjoint even when they share files.
            train_dataset, val_dataset = self._split_dataset(
                self._build_dataset_from_config(data_config, "data"),
                split,
            )
            test_dataset = self._build_dataset_from_config(
                self.config["testing"].get("data"), "testing"
            )
            return train_dataset, val_dataset, test_dataset

        data_config = self.config["data"]
        split = data_config.get("split", split)
        self._validate_split(split, expected_parts=3)
        return self._split_dataset(
            self._build_dataset_from_config(data_config, "data"),
            split,
        )

    def prepare_dataset(
        self,
        dataset: Dataset | None = None,
        split: list[float] = [0.8, 0.1, 0.1],
        train_dataset: torch.utils.data.Subset | None = None,
        val_dataset: torch.utils.data.Subset | None = None,
        test_dataset: torch.utils.data.Subset | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Create train, validation, and test datasets.

        Args:
            dataset (Dataset | None): Full dataset to split.
            split (list[float]): Fractions for train, validation, and test splits.
            train_dataset (torch.utils.data.Subset | None): Explicit training dataset.
            val_dataset (torch.utils.data.Subset | None): Explicit validation dataset.
            test_dataset (torch.utils.data.Subset | None): Explicit test dataset.

        Returns:
            tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.

        Raises:
            ValueError: If both a full dataset and split datasets are supplied.
            ValueError: If the split ratios are invalid.
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
            if train_dataset is None and val_dataset is None and test_dataset is None:
                return self._prepare_datasets_from_config(split)

            dataset = self._build_dataset_from_config(self.config.get("data"), "data")

        if train_dataset is None and val_dataset is None and test_dataset is None:
            split = self.config.get("data", {}).get("split", split)
            self._validate_split(split, expected_parts=3)
            train_dataset, val_dataset, test_dataset = self._split_dataset(
                dataset, split
            )

        return train_dataset, val_dataset, test_dataset

    def _loader_kwargs(self, section_name: str) -> dict[str, Any]:
        """Build keyword arguments for a loader section.

        Args:
            section_name (str): Config section name to read.

        Returns:
            dict[str, Any]: Keyword arguments for `torch_geometric.loader.DataLoader`.
        """
        section = self.config[section_name]
        return {
            "batch_size": section["batch_size"],
            "num_workers": section.get("num_workers", 0),
            "pin_memory": section.get("pin_memory", True),
            "drop_last": section.get("drop_last", False),
            "prefetch_factor": section.get("prefetch_factor", None),
            "persistent_workers": self.config["training"].get(
                "persistent_workers", False
            ),
            "shuffle": section.get("shuffle", True),
        }

    def prepare_dataloaders(
        self,
        dataset: Dataset | None = None,
        split: list[float] = [0.8, 0.1, 0.1],
        train_dataset: torch.utils.data.Subset | None = None,
        val_dataset: torch.utils.data.Subset | None = None,
        test_dataset: torch.utils.data.Subset | None = None,
        training_sampler: torch.utils.data.Sampler | None = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders.

        Args:
            dataset (Dataset | None): Full dataset to split.
            split (list[float]): Fractions for train, validation, and test splits.
            train_dataset (torch.utils.data.Subset | None): Explicit training dataset.
            val_dataset (torch.utils.data.Subset | None): Explicit validation dataset.
            test_dataset (torch.utils.data.Subset | None): Explicit test dataset.
            training_sampler (torch.utils.data.Sampler | None): Optional train sampler.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
        """
        train_dataset, val_dataset, test_dataset = self.prepare_dataset(
            dataset=dataset,
            split=split,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )

        train_kwargs = self._loader_kwargs("training")
        if training_sampler is not None:
            train_kwargs["shuffle"] = False

        train_loader = DataLoader(
            train_dataset,
            sampler=training_sampler,
            **train_kwargs,
        )
        val_loader = DataLoader(val_dataset, **self._loader_kwargs("validation"))
        test_loader = DataLoader(test_dataset, **self._loader_kwargs("testing"))

        self.logger.info(
            "Data loaders prepared with dataset sizes: %s, %s, %s",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
        return train_loader, val_loader, test_loader


class DistributedDataLoaderFactory(DataLoaderFactory):
    """Build distributed datasets and dataloaders from training configuration."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DistributedDataLoaderFactory Configuration",
        "type": "object",
        "properties": {
            "parallel": {
                "type": "object",
                "properties": {
                    "world_size": {"type": "integer", "minimum": 1},
                    "rank": {"type": "integer", "minimum": 0},
                    "output_device": {"type": ["integer", "null"]},
                    "find_unused_parameters": {"type": "boolean"},
                    "master_addr": {"type": "string"},
                    "master_port": {"type": "string"},
                },
                "required": ["world_size"],
                "additionalProperties": True,
            },
            "name": DataLoaderFactory.schema["properties"]["name"],
            "log_level": DataLoaderFactory.schema["properties"]["log_level"],
            "data": DATA_CONFIG_SCHEMA,
            "training": DataLoaderFactory.schema["properties"]["training"],
            "validation": DataLoaderFactory.schema["properties"]["validation"],
            "testing": DataLoaderFactory.schema["properties"]["testing"],
        },
        "required": ["parallel", "training", "validation", "testing"],
        "additionalProperties": True,
    }

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | None = None,
        rank: int | None = None,
    ) -> None:
        """Initialize the distributed data loader factory.

        Args:
            config (dict[str, Any]): Configuration used for dataset and loader setup.
            logger (logging.Logger | None): Optional logger to reuse.
            rank (int | None): Explicit rank override for distributed setup.
        """
        super().__init__(config=config, logger=logger)
        self.rank = config["parallel"].get("rank", 0) if rank is None else rank
        self.world_size = config["parallel"]["world_size"]
        self.train_sampler: torch.utils.data.DistributedSampler | None = None
        self.val_sampler: torch.utils.data.DistributedSampler | None = None
        self.test_sampler: torch.utils.data.DistributedSampler | None = None

    def prepare_dataloaders(
        self,
        dataset: Dataset | None = None,
        split: list[float] = [0.8, 0.1, 0.1],
        train_dataset: torch.utils.data.Dataset | torch.utils.data.Subset | None = None,
        val_dataset: torch.utils.data.Dataset | torch.utils.data.Subset | None = None,
        test_dataset: torch.utils.data.Dataset | torch.utils.data.Subset | None = None,
        training_sampler: torch.utils.data.Sampler | None = None,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create distributed train, validation, and test dataloaders.

        Args:
            dataset (Dataset | None): Full dataset to split.
            split (list[float]): Fractions for train, validation, and test splits.
            train_dataset (torch.utils.data.Dataset | torch.utils.data.Subset | None): Explicit training dataset.
            val_dataset (torch.utils.data.Dataset | torch.utils.data.Subset | None): Explicit validation dataset.
            test_dataset (torch.utils.data.Dataset | torch.utils.data.Subset | None): Explicit test dataset.
            training_sampler (torch.utils.data.Sampler | None): Optional train sampler override.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test dataloaders.
        """
        train_dataset, val_dataset, test_dataset = self.prepare_dataset(
            dataset=dataset,
            split=split,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )

        # Distributed samplers keep the data partitioning explicit in the DDP path.
        self.train_sampler = (
            training_sampler
            if training_sampler is not None
            else torch.utils.data.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        )
        self.val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        self.test_sampler = torch.utils.data.DistributedSampler(
            test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        train_kwargs = self._loader_kwargs("training")
        train_kwargs["shuffle"] = False
        val_kwargs = self._loader_kwargs("validation")
        val_kwargs["shuffle"] = False
        test_kwargs = self._loader_kwargs("testing")
        test_kwargs["shuffle"] = False

        train_loader = DataLoader(
            train_dataset,
            sampler=self.train_sampler,
            **train_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=self.val_sampler,
            **val_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            sampler=self.test_sampler,
            **test_kwargs,
        )

        self.logger.info(
            "Distributed data loaders prepared with dataset sizes: %s, %s, %s",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
        return train_loader, val_loader, test_loader
