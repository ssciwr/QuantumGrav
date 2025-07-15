# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import sys

# data handling
import juliacall as jcall

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass
from joblib import Parallel, delayed


@dataclass
class QGOntheflyConfig:
    """Config class for the QGDatasetOnthefly dataset. This class holds the configuration parameters for generating data on the fly using a Julia function."""

    atom_count_min: int = 1
    atom_count_max: int = 1
    seed: int = 42
    manifolds: list[int] = None
    boundaries: list[int] = None
    dimensions: list[int] = None
    n_samples: int = 64
    n_processes: int = -1
    julia_nthreads: int = 1


# FIXME: this is very unstable right now
class QGDatasetOnthefly(Dataset):
    """A dataset that generates data on the fly using a Julia function.

    Args:
        Dataset (Dataset): The base dataset class.
    """

    jl_generator: jcall.AnyValue | None = None
    jl_module: jcall.ModuleValue | None = None

    def __init__(
        self,
        config: QGOntheflyConfig,
        jl_code_path: str | Path | None = None,
        jl_func_name: str | None = None,
        jl_module_name: str | None = None,
        jl_base_module_path: str | Path | None = None,
        jl_dependencies: list[str] | None = None,
        transform: Callable[[dict[Any, Any]], Data] | None = None,
    ):
        """Initialize the dataset.

        Args:
            jl_code_path (str | Path | None, optional): The path to the Julia code file that contains the data generation function. Must be a module. Defaults to None.
            jl_func_name (str | None, optional): The name of the Julia function to call for data production. Defaults to None.
            jl_module_name (str | None, optional): The name of the Julia module to use. Defaults to None.

            transform (Callable[[dict[Any, Any]], Data] | None, optional): A function to transform the raw data. Defaults to None.

        Raises:
            ValueError: If the Julia code path is not provided.
            FileNotFoundError: _description_
        """

        if transform is None:
            self.transform = lambda x: Data.from_dict(x)
        else:
            self.transform = transform

        if jl_func_name is None:
            raise ValueError("Julia function name must be provided.")

        if jl_code_path is None:
            raise ValueError("Julia code path must be provided.")

        jl_code_path = Path(jl_code_path).resolve().absolute()
        if not jl_code_path.exists():
            raise FileNotFoundError(f"Julia code path {jl_code_path} does not exist.")

        if jl_module_name is None:
            jl_module_name = jl_code_path.stem

        try:
            self.jl_module = jcall.newmodule(jl_module_name)

            # add base module for dependencies
            if jl_base_module_path is None:
                raise NotImplementedError(
                    "Base module path must be provided at the moment"
                )
            else:
                self.jl_module.seval(
                    f'using Pkg; Pkg.develop(path="{jl_base_module_path}")'
                )  # only for now -> get from package index later

            # add dependencies if provided
            if jl_dependencies is not None:
                for dep in jl_dependencies:
                    self.jl_module.seval(f'using Pkg; Pkg.add("{dep}")')

            # load the julia data generation julia code
            self.jl_module.seval(f'push!(LOAD_PATH, "{jl_code_path}")')
            self.jl_module.seval("using QuantumGrav")
            self.jl_module.seval(f'include("{jl_code_path}")')
            # generate the julia object and call it later with arguments
            self.jl_generator = self.jl_module.seval(
                f"{jl_module_name}.{jl_func_name}({config.seed},{config.atom_count_min},{config.atom_count_max},{config.manifolds},{config.boundaries},{config.dimensions},{config.n_samples},{config.julia_nthreads > 1})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading Julia module {jl_module_name}: {e}"
            ) from e

        self.config = config
        self.databatch: list[Data] = []  # hold a batch of generated data

        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def len(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return sys.maxsize

    def get(self, _: int) -> Data:
        """Get a single data point. Generates a new datapoint on the fly upon each request and applies the transformation function to it before returning.

        Args:
            _ (int): Dummy argument to match the Dataset interface. does nothing here.

        Raises:
            RuntimeError: When the internally held julia module is not initialized.
            RuntimeError: When the data transformation fails.

        Yields:
            Data: The transformed data point.
        """
        if self.jl_module is None:
            raise RuntimeError("Julia module is not initialized.")

        if len(self.databatch) == 0:
            # Call the Julia function to get the data
            raw_data = self.jl_generator()

            try:
                # parallel processing in Julia is handled on the Julia side
                if self.config.n_processes != 1:
                    self.databatch = Parallel(n_jobs=self.config.n_processes)(
                        delayed(self.transform)(raw_datapoint)
                        for raw_datapoint in raw_data
                    )
                else:
                    self.databatch = [
                        self.transform(raw_datapoint) for raw_datapoint in raw_data
                    ]
            except Exception as e:
                self.logger.error(f"Error transforming data: {e}")
                raise RuntimeError(f"Error transforming data: {e}") from e

        datapoint = self.databatch.pop()

        yield datapoint
