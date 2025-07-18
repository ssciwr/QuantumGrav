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
            print("done")
        except Exception as e:
            raise RuntimeError(f"Error initializing Julia process: {e}") from e

        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def len(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return sys.maxsize

    def make_batch(self, size: int) -> list[Data]:
        return [self.get(i) for i in range(size)]

    def get(self, _: int) -> Data:
        # this breaks the contract of the 'get'method that the base class provides, but
        # it nevertheless is useful to generate training data
        if self.worker is None:
            raise RuntimeError("Worker process is not initialized.")

        if len(self.databatch) == 0:
            # Call the Julia function to get the data
            raw_data = [
                self.converter(x) for x in self.worker(self.config["batch_size"])
            ]
            if isinstance(raw_data, Exception):
                raise raw_data
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
            except Exception as e:
                raise RuntimeError(f"Error transforming data: {e}") from e

        datapoint = self.databatch.pop()

        return datapoint
