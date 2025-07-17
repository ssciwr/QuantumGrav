from . import julia_worker as jl_worker

# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import sys

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from typing import Any
from joblib import Parallel, delayed
from multiprocessing import Process, Pipe
import dill


class QGDatasetOnthefly(Dataset):
    """A dataset that generates data on the fly using a Julia function.

    Args:
        Dataset (Dataset): The base dataset class.
    """

    def __init__(
        self,
        config: dict[str, Any],
        jl_code_path: str | Path | None = None,
        jl_func_name: str | None = None,
        jl_module_name: str | None = None,
        jl_base_module_path: str | Path | None = None,
        jl_dependencies: list[str] | None = None,
        transform: Callable[[dict[Any, Any]], Data] | None = None,
    ):
        if transform is None:
            raise ValueError(
                "Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects."
            )
        self.transform = transform

        self.config = config
        self.databatch: list[Data] = []  # hold a batch of generated data
        self.parent_conn, self.child_conn = Pipe()

        try:
            self.worker = Process(
                target=jl_worker.worker_loop,
                args=(
                    self.child_conn,
                    config,
                    jl_code_path,
                    jl_func_name,
                    jl_module_name,
                    jl_base_module_path,
                    jl_dependencies,
                ),
            )
            self.worker.start()
            print("done")
        except Exception as e:
            raise RuntimeError(f"Error initializing Julia process: {e}") from e

        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def shutdown(self):
        if (
            self.worker is not None
            and self.parent_conn is not None
            and self.child_conn is not None
        ):
            self.parent_conn.send("STOP")
            self.worker.join()
            self.parent_conn.close()
            self.child_conn.close()
            self.parent_conn = None
            self.child_conn = None
            self.worker = None

    def __del__(self):
        """Ensure the worker process is terminated when the dataset is deleted."""
        try:
            self.shutdown()
        except Exception as e:
            print(f"Error shutting down worker: {e}")

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
            self.parent_conn.send("GET")
            raw_bytes = self.parent_conn.recv()
            raw_data = dill.loads(raw_bytes)
            if isinstance(raw_data, Exception):
                raise raw_data
            try:
                # parallel processing in Julia is handled on the Julia side
                # use primitve indexing here to avoid issues with julia arrays
                if self.config["n_processes"] > 1:
                    self.databatch = Parallel(n_jobs=self.config["n_processes"])(
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
