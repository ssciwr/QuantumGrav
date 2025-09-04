# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import torch

# data handling
import h5py
import zarr

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable, Collection

# internals
from .dataset_base import QGDatasetBase


class QGDataset(QGDatasetBase, Dataset):
    """A dataset class for QuantumGrav data that is designed to handle large datasets stored on disk. This class provides methods for loading, processing, and writing data that are common to both in-memory and on-disk datasets."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        mode: str = "hdf5",
        reader: Callable[
            [h5py.File | zarr.Group, torch.dtype, torch.dtype, bool], list[Data]
        ]
        | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        chunksize: int = 1000,
        n_processes: int = 1,
        # dataset properties
        transform: Callable[[Data | Collection], Data] | None = None,
        pre_transform: Callable[[Data | Collection], Data] | None = None,
        pre_filter: Callable[[Data | Collection], bool] | None = None,
    ):
        """Create a new QGDataset instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets that are stored on disk.

        Args:
            input (list[str  |  Path] | Callable[[Any], dict]): List of input hdf5 file paths.
            output (str | Path): Output directory where processed data will be stored.
            mode (str): File storage mode. 'zarr' or 'hdf5'
            reader (Callable[[h5py.File | zarr.Group, int], list[Data]] | None, optional): Function to read data from the hdf5 file. Defaults to None.
            float_type (torch.dtype, optional): Data type for float tensors. Defaults to torch.float32.
            int_type (torch.dtype, optional): Data type for int tensors. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data. Defaults to True.
            chunksize (int, optional): Size of data chunks to process at once. Defaults to 1000.
            n_processes (int, optional): Number of processes to use for data loading. Defaults to 1.
            transform (Callable[[Data], Data] | None, optional): Function to transform the data. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): Function to pre-transform the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): Function to pre-filter the data. Defaults to None.
        """

        QGDatasetBase.__init__(
            self,
            input,
            output,
            mode=mode,
            reader=reader,
            float_type=float_type,
            int_type=int_type,
            validate_data=validate_data,
            chunksize=chunksize,
            n_processes=n_processes,
        )

        Dataset.__init__(
            self,
            root=output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def write_data(self, data: list[Data], idx: int) -> int:
        """Write the processed data to disk using `torch.save`. This is a default implementation that can be overridden by subclasses, and is intended to be used in the data loading pipeline. Thus, is not intended to be called directly.

        Args:
            data (list[Data]): The list of Data objects to write to disk.
            idx (int): The index to use for naming the files.
        """
        if not Path(self.processed_dir).exists():
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        for d in data:
            if d is not None:
                file_path = Path(self.processed_dir) / f"data_{idx}.pt"
                torch.save(d, file_path)
                idx += 1
        return idx

    def process(self) -> None:
        """Process the dataset from the read rawdata into its final form."""
        # process data files
        k = 0  # index to create the filenames for the processed data
        for file in self.input:
            if self.mode == "hdf5":
                raw_file = h5py.File(str(Path(file).resolve().absolute()), "r")
                num_chunks = raw_file["num_causal_sets"][()] // self.chunksize

            else:
                raw_file = zarr.storage.LocalStore(
                    str(Path(file).resolve().absolute()), read_only=True
                )
                root = zarr.open_group(raw_file, path="", mode="r")
                N = int(root["num_samples"][0])
                num_chunks = N // self.chunksize

            for i in range(0, num_chunks * self.chunksize, self.chunksize):
                data = self.process_chunk(
                    raw_file,
                    i,
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                )

                k = self.write_data(data, k)

            # final chunk processing
            data = self.process_chunk(
                raw_file,
                num_chunks * self.chunksize,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
            )

            k = self.write_data(data, k)

            raw_file.close()

    def get(self, idx: int) -> Data:
        """Get a single data sample by index."""
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        if idx < 0 or idx >= self._num_samples:
            raise IndexError("Index out of bounds.")

        # Load the data from the processed files
        datapoint = torch.load(
            Path(self.processed_dir) / f"data_{idx}.pt", weights_only=False
        )
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint

    def len(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.processed_file_names)
