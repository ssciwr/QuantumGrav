# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import sys

# data handling
import juliacall as jcall

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

    jl_generator: jcall.AnyValue | None = None
    jl_module: jcall.ModuleValue | None = None

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
        """_summary_

        Args:
            config (dict[str, Any]): _description_
            jl_code_path (str | Path | None, optional): _description_. Defaults to None.
            jl_func_name (str | None, optional): _description_. Defaults to None.
            jl_module_name (str | None, optional): _description_. Defaults to None.
            jl_base_module_path (str | Path | None, optional): _description_. Defaults to None.
            jl_dependencies (list[str] | None, optional): _description_. Defaults to None.
            transform (Callable[[dict[Any, Any]], Data] | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            FileNotFoundError: _description_
            NotImplementedError: _description_
            RuntimeError: _description_
        """

        if transform is None:
            raise ValueError(
                "Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects."
            )
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

            # add dependencies if provided\
            if jl_dependencies is not None:
                for dep in jl_dependencies:
                    self.jl_module.seval(f'using Pkg; Pkg.add("{dep}")')

            # load the julia data generation julia code
            self.jl_module.seval(f'push!(LOAD_PATH, "{jl_code_path}")')
            self.jl_module.seval("using QuantumGrav")
            self.jl_module.seval(f'include("{jl_code_path}")')
            # generate the julia object and call it later with arguments
            # self.jl_generator = self.jl_module.seval(
            # f"{jl_module_name}.{jl_func_name}(config)"
            # )

            generator_constructor = getattr(self.jl_module, jl_func_name)

            # print("fucking dogshit julia dict: ", jl_config)
            self.jl_generator = generator_constructor(config)
        except Exception as e:
            raise RuntimeError(
                f"Error loading Julia module {jl_module_name}: {e}"
            ) from e

        self.config = config
        self.databatch: list[Data] = []  # hold a batch of generated data

        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def _convert_to_julia_dict(self, py_dict: dict[str, Any]) -> jcall.AnyValue:
        """Convert a Python dictionary to a Julia dictionary.

        Args:
            config (dict[str, Any]): The Python dictionary to convert.

        Returns:
            jcall.AnyValue: The converted Julia dictionary.
        """
        jlstore = self.jl_module.seval("(k, v) -> (@eval $(Symbol(k)) = $v; return)")
        jlstore("config_dict", py_dict)

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
                if self.config["n_processes"] != 1:
                    self.databatch = Parallel(n_jobs=self.config["n_processes"])(
                        delayed(self.transform)(raw_datapoint)
                        for raw_datapoint in raw_data
                    )
                else:
                    self.databatch = [
                        self.transform(raw_datapoint) for raw_datapoint in raw_data
                    ]
            except Exception as e:
                raise RuntimeError(f"Error transforming data: {e}") from e

        datapoint = self.databatch.pop()

        return datapoint
