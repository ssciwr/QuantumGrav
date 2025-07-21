# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import torch

# data handling
import h5py

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from typing import Any

# internals
from .dataset_base import QGDatasetBase


class QGDataset(QGDatasetBase, Dataset):
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


        """

        QGDatasetBase.__init__(
            input,
            output,
            get_metadata,
            reader,
            float_type,
            int_type,
            validate_data,
            chunksize=chunksize,
            n_processes=n_processes,
        )

        Dataset.__init__(
            output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Get the names of the raw files in the dataset.

        Returns:
            list[str]: A list of raw file names.
        """
        return super().raw_file_names

    @property
    def processed_paths(self) -> list[str]:
        """Get the names of the processed files in the dataset.

        Returns:
            list[str]: A list of processed file names.
        """
        return super().processed_paths

    def write_data(self, data: list[Data]) -> None:
        """Write the processed data to disk using `torch.save`. This is a default implementation that can be overridden by subclasses, and is intended to be used in the data loading pipeline. Thus, is not intended to be called directly.

        Args:
            data (list[Data]): The list of Data objects to write to disk.
        """
        if not Path(self.processed_dir).exists():
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        for i, d in enumerate(data):
            if d is not None:
                file_path = Path(self.processed_dir) / f"data_{i}.pt"
                torch.save(d, file_path)

    def load_data(self, indices) -> list[Data]:
        """Load a list of Data objects from disk.

        Args:
            indices (list[int]): A list of indices to load.

        Returns:
            list[Data]: A list of loaded Data objects.
        """
        data = []
        for i in indices:
            file_path = Path(self.processed_dir) / f"data_{i}.pt"
            if file_path.exists():
                datapoint = torch.load(file_path, weights_only=False)
                data.append(datapoint)
        return data

    def process(self) -> None:
        """Process the dataset from the read rawdata into its final form."""
        # process data files
        for file in self.raw_file_names:
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

                # final chunk processing
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
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        super().len()

    def get(self, idx: int) -> Data:
        """Get a single data sample by index."""
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        if idx < 0 or idx >= self._num_samples:
            raise IndexError("Index out of bounds.")

        # Load the data from the processed files
        datapoint = torch.load(Path(self.processed_dir) / f"data_{idx}.pt")
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint
