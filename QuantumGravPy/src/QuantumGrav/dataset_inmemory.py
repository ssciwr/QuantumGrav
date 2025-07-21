# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset
import torch

# data handling
import h5py

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from typing import Any

# internals
from .dataset_base import QGDatasetBase


class QGDatasetInMemory(QGDatasetBase, InMemoryDataset):
    """A dataset class for QuantumGrav data that can be loaded into memory."""

    def __init__(
        self,
        input: list[str | Path] | Callable[[Any], dict],
        output: str | Path,
        reader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        get_metadata: Callable[[str | Path], dict] | None = None,
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
            input (list[str  |  Path] | Callable[[Any], dict]): _description_
            output (str | Path): _description_
            reader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): _description_. Defaults to None.
            get_metadata (Callable[[str  |  Path], dict] | None, optional): _description_. Defaults to None.
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
            self,
            input,
            output,
            get_metadata=get_metadata,
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
        print("Processing dataset...")
        data_list = []
        for file in self.input:
            with h5py.File(str(Path(file).resolve().absolute()), "r") as raw_file:
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
                        pre_transform=self.pre_transform,
                        pre_filter=self.pre_filter,
                    )

                    data_list.extend(data)

                # final chunk processing
                for i in range(
                    num_chunks * self.chunksize, final_chunk, self.chunksize
                ):
                    data = self.process_chunk(
                        raw_file,
                        i,
                        final_chunk,
                        self.n_processes,
                        self.pre_transform,
                        self.pre_filter,
                    )

                    data_list.extend(data)

        InMemoryDataset.save(data_list, Path(self.processed_dir) / "data.pt")
