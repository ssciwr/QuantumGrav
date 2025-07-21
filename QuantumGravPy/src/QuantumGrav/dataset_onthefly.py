import juliacall as jcall
from . import julia_worker as jl_worker

# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import sys

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from typing import Any
from joblib import Parallel, delayed


class QGDatasetOnthefly(Dataset):
    """A dataset that generates data on the fly using a Julia function.

    Args:
        Dataset (Dataset): The base dataset class.
    """

    def __init__(
        self,
        config: dict[str, Any],
        jl_code_path: str | Path | None = None,
        jl_constructor_name: str | None = None,
        jl_base_module_path: str | Path | None = None,
        jl_dependencies: list[str] | None = None,
        transform: Callable[[dict[Any, Any]], Data] | None = None,
        converter: Callable[[Any], Any] | None = None,
    ):
        """Initialize the dataset.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            jl_code_path (str | Path | None, optional): Path to the Julia code. Defaults to None.
            jl_constructor_name (str | None, optional): Name of the Julia constructor. Defaults to None.
            jl_base_module_path (str | Path | None, optional): Path to the base Julia module. Defaults to None.
            jl_dependencies (list[str] | None, optional): List of Julia dependencies. Defaults to None.
            transform (Callable[[dict[Any, Any]], Data] | None, optional): Function to transform raw data into PyTorch Geometric Data objects. Defaults to None.
            converter (Callable[[Any], Any] | None, optional): Function to convert Julia objects into standard Python objects. Defaults to None.

        Raises:
            ValueError: If the transform function is not provided.
            ValueError: If the converter function is not provided.
            RuntimeError: If there is an error initializing the Julia process.
        """
        if transform is None:
            raise ValueError(
                "Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects."
            )
        self.transform = transform

        if converter is None:
            raise ValueError(
                "Converter function must be provided to convert Julia objects into standard Python objects."
            )
        else:
            self.converter = converter

        self.config = config
        self.databatch: list[Data] = []  # hold a batch of generated data

        try:
            self.worker = jl_worker.JuliaWorker(
                config,
                jl_code_path,
                jl_constructor_name,
                jl_base_module_path,
                jl_dependencies,
            )
        except jcall.JuliaError as e:
            raise RuntimeError(f"Error initializing Julia worker: {e}") from e
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(f"Path to file or directory not found: {e}") from e
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Invalid configuration for Julia worker: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected exception while initializing Julia worker: {e}"
            ) from e

        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def len(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return sys.maxsize

    def make_batch(self, size: int) -> list[Data]:
        """Create a batch of data.

        Args:
            size (int): The number of samples in the batch.

        Returns:
            list[Data]: The list of Data objects in the batch.
        """
        return [self.get(i) for i in range(size)]

    def get(self, _: int) -> Data:
        """Get a data point from the dataset.

        Args:
            _ (int): The index of the data point to retrieve. This is ignored since the dataset generates data on the fly.

        Raises:
            RuntimeError: If the worker process is not initialized.
            raw_data: If there is an error retrieving raw data from the worker.
            RuntimeError: If there is an error transforming the data.

        Returns:
            Data: The transformed data point.
        """
        # this breaks the contract of the 'get'method that the base class provides, but
        # it nevertheless is useful to generate training data
        if self.worker is None:
            raise RuntimeError("Worker process is not initialized.")

        if len(self.databatch) == 0:
            # Call the Julia function to get the data
            try:
                raw_data = [
                    self.converter(x) for x in self.worker(self.config["batch_size"])
                ]
            except jcall.JuliaError as e:
                raise RuntimeError(f"Julia worker failed to generate data: {e}") from e
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Invalid configuration or data access: {e}") from e

            if isinstance(raw_data, Exception):
                raise RuntimeError(
                    "Unexpected error in data generation or conversion"
                ) from raw_data
            try:
                # parallel processing in Julia is handled on the Julia side
                # use primitve indexing here to avoid issues with julia arrays
                if self.config["n_processes"] > 1:
                    self.databatch = Parallel(
                        n_jobs=self.config["n_processes"], verbose=10
                    )(
                        delayed(self.transform)(raw_data[i])
                        for i in range(len(raw_data))
                    )
                else:
                    self.databatch = [
                        self.transform(raw_data[i]) for i in range(len(raw_data))
                    ]
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(f"Data transformation failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error transforming data: {e}") from e

        datapoint = self.databatch.pop()

        return datapoint
