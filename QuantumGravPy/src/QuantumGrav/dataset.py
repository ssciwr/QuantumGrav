# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch
import sys

# data handling
import h5py
import json
import logging
import juliacall as jcall

# system imports and quality of life tools
from pathlib import Path
import os
from collections.abc import Callable
from typing import Any
from joblib import Parallel, delayed

# julia interface


class QGDatasetMixin:
    """Mixin class for QG dataset handling. Provides methods for loading, processing, and writing data that are common to both in-memory and on-disk datasets."""

    def __init__(
        self,
        input: list[str | Path],
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        """Initialize a DatasetMixin instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets. It provides a common interface for both in-memory and on-disk datasets. It is not to be instantiated directly, but rather used as a mixin for other dataset classes.

        Args:
            input (list[str  |  Path] : The list of input files for the dataset, or a callable that generates a set of input files.
            get_metadata (Callable[[str  |  Path], dict] | None, optional): A function to retrieve metadata for the dataset. Defaults to None.
            loader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from a file. Defaults to None.
            writer (Callable[[list[Data], str, dict[Any, Any]], None] | None, optional): A function to write data to a file. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point values. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integer values. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after loading. Defaults to True.
            parallel_processing (bool, optional): Whether to use parallel processing for data loading. Defaults to False.
            writer_kwargs (dict[str, Any], optional): Additional keyword arguments to pass to the writer function. Defaults to None.

        Raises:
            ValueError: If one of the input data files is not a valid HDF5 file
            ValueError: If the metadata retrieval function is invalid.
            FileNotFoundError: If an input file does not exist.
        """
        self.writer_kwargs = writer_kwargs or {}
        if loader is None:
            raise ValueError("A loader function must be provided to load the data.")

        if get_metadata is None:
            raise ValueError("A metadata retrieval function must be provided.")

        self.input = input
        self._num_samples = None
        self.data_loader = loader
        self.data_writer = writer
        self.get_metadata = get_metadata
        self.metadata = {}
        self.float_type = float_type
        self.int_type = int_type
        self.validate_data = validate_data
        self.parallel_processing = parallel_processing

        # ensure the input is a list of paths
        if self.processed_dir is not None:
            with open(os.path.join(self.processed_dir, "metadata.json"), "r") as f:
                self.metadata = json.load(f)

        # get the number of samples in the dataset
        self._num_samples = 0
        for file in self.input:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist.")
            with h5py.File(file, "r") as f:
                self._num_samples += f["num_causal_sets"][()]

    @property
    def processed_dir(self) -> str | None:
        """Get the path to the processed directory.

        Returns:
            str: The path to the processed directory, or None if it doesn't exist.
        """
        processed_path = os.path.join(self.root, "processed")
        if not os.path.exists(processed_path):
            return None
        return processed_path

    @property
    def processed_files(self) -> list[str]:
        """Get a list of processed files in the processed directory.

        Returns:
            list[str]: A list of processed file paths, excluding JSON files.
        """

        if not os.path.isdir(self.processed_dir):
            return []

        return [
            os.path.join(self.processed_dir, f)
            for f in os.listdir(self.processed_dir)
            if f.endswith(".json") is False
        ]

    @property
    def raw_file_names(self) -> list[str]:
        """Get the raw file names from the input list.

        Returns:
            list[str]: A list of raw file names.
        """
        return [os.path.basename(f) for f in self.input if Path(f).suffix == ".h5"]

    @property
    def processed_file_names(self) -> list[str]:
        """Get the processed file names from the processed directory.

        Returns:
            list[str]: A list of processed file names.
        """
        if os.path.isdir(self.root) is False:
            return []

        return [f for f in os.listdir(self.root) if f.endswith(".pt")]

    def process_chunk(
        self,
        raw_file: h5py.File,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> tuple[int, list[Data]]:
        """Process a chunk of data from the raw HDF5 file.

        Args:
            raw_file (h5py.File): The raw HDF5 file to process data from.

        Returns:
            list[Data]: A list of processed data items.
        """
        data_list = self.data_loader(
            raw_file,
            float_type=self.float_type,
            int_type=self.int_type,
            validate=self.validate_data,
        )

        if self.parallel_processing:
            processed = Parallel(n_jobs=-1)(
                delayed(pre_transform)(data_point)
                for data_point in data_list
                if delayed(pre_filter)(data_point)
            )
        else:
            processed = [
                pre_transform(data_point)
                for data_point in data_list
                if pre_filter(data_point)
            ]
        return len(data_list), processed


class QGDatasetInMemory(QGDatasetMixin, InMemoryDataset):
    """A dataset class for QuantumGrav data that can be loaded into memory."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        """Initialize a QGDatasetInMemory instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets that can be loaded into memory completely.

        Args:
            input (list[str  |  Path] | Callable[[Any], dict]): The list of input files for the dataset, or a callable that generates a set of input files.
            output (str | Path): The output directory for processed data.
            transform (Callable[[Data], Data] | None, optional): A function to apply transformations to the data. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): A function to apply preprocessing transformations to the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function to filter the data before processing. Defaults to None.
            get_metadata (Callable[[str  |  Path], dict] | None, optional): A function to retrieve metadata for the dataset. Defaults to None.
            loader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from raw HDF5 files. Defaults to None.
            writer (Callable[[list[Data], str, dict[Any, Any]], None] | None, optional): A function to write processed data to disk. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point values. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integer values. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after processing. Defaults to True.
            parallel_processing (bool, optional): Whether to use parallel processing for data loading and processing. Defaults to False.
            writer_kwargs (dict[str, Any], optional): Additional keyword arguments to pass to the writer function. Defaults to None.
        """
        QGDatasetMixin.__init__(
            input,
            get_metadata,
            loader,
            writer,
            float_type,
            int_type,
            validate_data,
            parallel_processing,
            writer_kwargs,
        )

        InMemoryDataset.__init__(
            output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def process(self) -> None:
        """Process the dataset by reading raw data files, applying transformations, and saving processed data."""
        if (
            not os.path.exists(self.processed_dir)
            or len(self.processed_file_names) == 0
        ):
            os.makedirs(self.processed_dir, exist_ok=True)

            self.metadata = self.get_metadata(self.input)
            with open(os.path.join(self.processed_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)

            full_data = []
            for file in self.raw_paths:
                with h5py.File(file, "r") as raw_file:
                    num_read = 0
                    # read the data in chunks and process it parallelized or
                    # sequentially based on the parallel_processing flag
                    while num_read < raw_file["num_causal_sets"][()]:
                        read_raw, processed = self.process_chunk(
                            raw_file,
                            pre_transform=self.pre_transform,
                            pre_filter=self.pre_filter,
                        )
                        full_data.extend(processed)
                        num_read += read_raw

            # if there is no separate data writer given,
            # use the default save method of InMemoryDataset
            if self.data_writer is None:
                self.save(full_data, self.processed_dir, self.writer_kwargs)
            else:
                self.data_writer(
                    full_data,
                    self.processed_dir,
                    self.writer_kwargs,
                )


class QGDataset(QGDatasetMixin, Dataset):
    """A dataset class for QuantumGrav data that is designed to handle large datasets stored on disk. This class provides methods for loading, processing, and writing data that are common to both in-memory and on-disk datasets."""

    def __init__(
        self,
        input: list[str | Path] | Callable[[Any], dict],
        output: str | Path,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        reader: Callable[[str | Path, int], Data] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        """Initialize a QGDataset instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets that are stored on disk. It provides a common interface for both in-memory and on-disk datasets.

        Args:
            input (list[str  |  Path] | Callable[[Any], dict]): The input data source, either a list of file paths or a function that returns a dictionary of data.
            output (str | Path): The output directory where processed data will be saved.
            transform (Callable[[Data], Data] | None, optional): A function to transform the data after loading. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): A function to transform the data before loading. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function to filter the data before loading. Defaults to None.
            get_metadata (Callable[[str  |  Path], dict] | None, optional): A function to get metadata from the input files. Defaults to None.
            loader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from raw HDF5 files. Defaults to None.
            writer (Callable[[list[Data], str, dict[Any, Any]], None] | None, optional): A function to write processed data to disk. Defaults to None.
            reader (Callable[[str  |  Path, int], Data] | None, optional): A function to read processed data from disk. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point numbers. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integers. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after loading. Defaults to True.
            parallel_processing (bool, optional): Whether to use parallel processing for data loading and processing. Defaults to False.
            writer_kwargs (dict[str, Any], optional): Additional keyword arguments to pass to the writer function. Defaults to None.

        Raises:
            ValueError: If the input data source is invalid.
            ValueError: If the output directory is invalid.
        """
        if writer is None:
            raise ValueError("A writer function must be provided to save the data.")

        if reader is None:
            raise ValueError("A reader function must be provided to read the data.")

        QGDatasetMixin.__init__(
            input,
            get_metadata,
            loader,
            writer,
            float_type,
            int_type,
            validate_data,
            parallel_processing,
            writer_kwargs,
        )

        Dataset.__init__(
            output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.reader = reader

        if self.processed_dir is not None:
            self.cached_processed_files = self.processed_files
        else:
            self.cached_processed_files = []

    def process(self) -> None:
        """Process the dataset and save the processed data to disk."""
        if (
            not os.path.exists(self.processed_dir)
            or len(self.processed_file_names) == 0
        ):
            os.makedirs(self.processed_dir, exist_ok=True)

            self.metadata = self.get_metadata(self.input)
            with open(os.path.join(self.processed_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)

            for file in self.raw_paths:
                with h5py.File(file, "r") as raw_file:
                    num_read = 0
                    # read the data in chunks and process it parallelized or
                    # sequentially based on the parallel_processing flag
                    while num_read < raw_file["num_causal_sets"][()]:
                        read_raw, processed = self.process_chunk(
                            raw_file,
                            pre_transform=self.pre_transform,
                            pre_filter=self.pre_filter,
                        )

                        num_read += read_raw

                        self.data_writer(
                            processed,
                            self.processed_dir,
                            self.writer_kwargs,
                        )

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_samples

    def get(self, idx: int) -> Data:
        """Get a single data point by index."""
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        if idx < 0 or idx >= self._num_samples:
            raise IndexError("Index out of bounds.")

        # Load the data from the processed files
        data = self.reader(os.path.join(self.processed_dir), idx)
        if self.transform is not None:
            data = self.transform(data)
        return data


class QGDatasetOnthefly(Dataset):
    def __init__(
        self,
        jl_code_path: str | Path,
        jl_func_name: str,
        jl_module_name: str | None = None,
        jl_log_level: int = logging.INFO,
        transform: Callable[[dict[Any, Any]], Data] | None = None,
    ):
        self.jl = jcall.newmodule("GenerateData")
        self.add_to_jl_load_path(jl_code_path)
        self.func_name = jl_func_name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(jl_log_level)
        if transform is None:
            self.transform = lambda x: Data.from_dict(x)
        else:
            self.transform = transform

        self.jl.seval(f'include("{jl_code_path}")')
        if jl_module_name is not None:
            self.jl.seval(f"using {jl_module_name}")
        super().__init__(None, transform=transform, pre_transform=None, pre_filter=None)

    def add_to_jl_load_path(self, path: str | Path) -> None:
        self.jl.seval(f'push!(LOAD_PATH, "{path}")')

    def len(self) -> int:
        return sys.maxsize

    def get(self, idx: int) -> Data:
        """Get a single data point by index."""
        if self.jl.module is None:
            raise RuntimeError("Julia module is not initialized.")

        # Call the Julia function to get the data
        raw_data = self.jl.module.seval(f"{self.func_name}()")

        try:
            data = self.transform(raw_data)
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            raise RuntimeError(f"Error transforming data: {e}") from e

        return data
