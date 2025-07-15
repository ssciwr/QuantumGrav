# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import torch

# data handling
import h5py
import json

# system imports and quality of life tools
from pathlib import Path
import os
from collections.abc import Callable
from typing import Any

# internals
from .dataset_mixin import QGDatasetMixin


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
