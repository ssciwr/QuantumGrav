# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset
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
            output,
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

    # TODO: I think there is some stuff missing here.
