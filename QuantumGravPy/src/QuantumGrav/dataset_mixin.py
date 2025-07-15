# pytorch and torch geometric imports
from torch_geometric.data import Data
import torch

# data handling
import h5py
import json

# system imports and quality of life tools
from pathlib import Path
import os
from collections.abc import Callable
from typing import Any
from joblib import Parallel, delayed


class QGDatasetMixin:
    """Mixin class that provides common functionality for the dataset classes. Works only for file-based datasets. Provides methods for loading, processing, and writing data that are common to both in-memory and on-disk datasets."""

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
