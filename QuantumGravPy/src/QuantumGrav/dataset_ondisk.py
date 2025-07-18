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
        get_metadata: Callable[[str | Path], dict] | None = None,
        reader: Callable[[h5py.File, int], list[Data]] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        chunksize: int = 1000,
        n_processes: int = 1,
        # dataset properties
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ):
        """_summary_

        Args:
            input (list[str  |  Path] | Callable[[Any], dict]): _description_
            output (str | Path): _description_
            get_metadata (Callable[[str  |  Path], dict] | None, optional): _description_. Defaults to None.
            reader (Callable[[h5py.File, int], list[Data]] | None, optional): _description_. Defaults to None.
            float_type (torch.dtype, optional): _description_. Defaults to torch.float32.
            int_type (torch.dtype, optional): _description_. Defaults to torch.int64.
            validate_data (bool, optional): _description_. Defaults to True.
            chunksize (int, optional): _description_. Defaults to 1000.
            n_processes (int, optional): _description_. Defaults to 1.
            transform (Callable[[Data], Data] | None, optional): _description_. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): _description_. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if reader is None:
            raise ValueError("A reader function must be provided to read the data.")
        self.chunksize = chunksize
        self.n_processes = n_processes
        QGDatasetMixin.__init__(
            input,
            get_metadata,
            reader,
            float_type,
            int_type,
            validate_data,
        )

        Dataset.__init__(
            output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def process(self) -> None:
        """_summary_"""
        if (
            not os.path.exists(self.processed_dir)
            or len(self.processed_file_names) == 0
        ):
            os.makedirs(self.processed_dir, exist_ok=True)

            self.metadata = self.get_metadata(self.input)
            with open(os.path.join(self.processed_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)

        # process data files
        for file in self.raw_paths:
            with h5py.File(file, "r") as raw_file:
                # read the data in chunks and process it parallelized or
                # sequentially based on the parallel_processing flag
                num_chunks = raw_file["num_causal_sets"][()] % self.chunksize
                final_chunk = (
                    raw_file["num_causal_sets"][()] - num_chunks * self.chunksize
                )

                for i in range(0, num_chunks * self.chunksize, self.chunksize):
                    data = self.process_chunk(
                        raw_file,
                        i,
                        self.chunksize,
                        self.num_processes,
                        self.pre_transform,
                        self.pre_filter,
                    )

                    self.write_data(data)

                for i in range(
                    num_chunks * self.chunksize, final_chunk, self.chunksize
                ):
                    data = self.process_chunk(
                        raw_file,
                        i,
                        final_chunk,
                        self.num_processes,
                        self.pre_transform,
                        self.pre_filter,
                    )

                    self.write_data(data)

    def len(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._num_samples

    def get(self, idx: int) -> Data:
        """_summary_

        Args:
            idx (int): _description_

        Raises:
            ValueError: _description_
            IndexError: _description_

        Returns:
            Data: _description_
        """
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        if idx < 0 or idx >= self._num_samples:
            raise IndexError("Index out of bounds.")

        # Load the data from the processed files
        datapoint = torch.load(Path(self.processed_dir) / f"data_{idx}.pt")
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint
