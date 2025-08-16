# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset
import torch

# data handling
import h5py
import zarr

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable

# internals
from .dataset_base import QGDatasetBase


class QGDatasetInMemory(QGDatasetBase, InMemoryDataset):
    """A dataset class for QuantumGrav data that can be loaded into memory."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        mode: str = "hdf5",
        reader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
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
        """Initialize a QGDatasetInMemory instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets that can be loaded into memory completely.

        Args:
            input (list[str  |  Path]): A list of file paths (as strings or Path objects) to the input data files.
            output (str | Path): A file path (as a string or Path object) to the output data file.
            mode (str): File storage mode. 'zarr' or 'hdf5'
            reader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to read the data from the input files. Defaults to None.
            float_type (torch.dtype, optional): Data type for float tensors. Defaults to torch.float32.
            int_type (torch.dtype, optional): Data type for int tensors. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data. Defaults to True.
            chunksize (int, optional): Size of data chunks to process at once. Defaults to 1000.
            n_processes (int, optional): Number of processes to use for data loading. Defaults to 1.
            transform (Callable[[Data], Data] | None, optional): Function to transform the data each time the data is loaded. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): Function to transform the read data once and store the results on the disk. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): Function to pre-filter the data once and store the results on the disk. Defaults to None.
        """
        QGDatasetBase.__init__(
            self,
            input,
            output,
            mode,
            reader=reader,
            float_type=float_type,
            int_type=int_type,
            validate_data=validate_data,
            chunksize=chunksize,
            n_processes=n_processes,
        )

        InMemoryDataset.__init__(
            self,
            root=output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.load(str(Path(self.processed_dir) / "data.pt"))

    def process(self) -> None:
        """Process the dataset from the read rawdata into its final form."""

        data_list = []

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

            # read the data in chunks and process it parallelized or
            # sequentially based on the parallel_processing flag

            for i in range(0, num_chunks * self.chunksize, self.chunksize):
                data = self.process_chunk(
                    raw_file,
                    i,
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                )

                data_list.extend(data)

            # final chunk processing
            data = self.process_chunk(
                raw_file,
                num_chunks * self.chunksize,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
            )

            data_list.extend(data)

            raw_file.close()

        InMemoryDataset.save(data_list, Path(self.processed_dir) / "data.pt")
