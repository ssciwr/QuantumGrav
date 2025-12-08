# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import torch

# data handling
import zarr

# system imports and quality of life tools
from pathlib import Path

from collections.abc import Callable, Sequence, Collection
from typing import Any, Tuple

# internals
from .dataset_base import QGDatasetBase


class QGDataset(QGDatasetBase, Dataset):
    """A dataset class for QuantumGrav data that is designed to handle large datasets stored on disk. This class provides methods for loading, processing, and writing data that are common to both in-memory and on-disk datasets."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        reader: Callable[
            [zarr.Group, int, torch.dtype, torch.dtype, bool], Collection[Any]
        ]
        | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        chunksize: int = 1000,
        n_processes: int = 1,
        # dataset properties
        transform: Callable[[Data | Collection[Any]], Data] | None = None,
        pre_transform: Callable[[Data | Collection[Any]], Data] | None = None,
        pre_filter: Callable[[Data | Collection[Any]], bool] | None = None,
    ):
        """Create a new QGDataset instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets that are stored on disk. When there is no pre_transform and no pre_filter is given, the system will not create a `processed` directory.

        Args:
            input (list[str  |  Path] | Callable[[Any], dict]): List of input zarr file paths.
            output (str | Path): Output directory where processed data will be stored.
            reader (Callable[[zarr.Group, int], list[Data]] | None, optional): Function to read data from the zarr file. Defaults to None.
            float_type (torch.dtype, optional): Data type for float tensors. Defaults to torch.float32.
            int_type (torch.dtype, optional): Data type for int tensors. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data. Defaults to True.
            chunksize (int, optional): Size of data chunks to process at once. Defaults to 1000.
            n_processes (int, optional): Number of processes to use for data loading. Defaults to 1.
            transform (Callable[[Data], Data] | None, optional): Function to transform the data. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): Function to pre-transform the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): Function to pre-filter the data. Defaults to None.
        """
        preprocess = pre_transform is not None or pre_filter is not None
        QGDatasetBase.__init__(
            self,
            input,
            output,
            reader=reader,
            float_type=float_type,
            int_type=int_type,
            validate_data=validate_data,
            chunksize=chunksize,
            n_processes=n_processes,
            preprocess=preprocess,
        )

        Dataset.__init__(
            self,
            root=output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.stores = {}

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
        if self.pre_filter is None and self.pre_transform is None:
            return

        for file in self.input:
            N = self._get_num_samples_per_file(Path(file).resolve().absolute())

            num_chunks = N // self.chunksize

            raw_file = zarr.storage.LocalStore(
                str(Path(file).resolve().absolute()), read_only=True
            )

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

    def _get_store_group(
        self, file: Path | str
    ) -> Tuple[zarr.storage.LocalStore, zarr.Group]:
        """Get a requested open store and add it to an internal cache if not open yet.

        Args:
            file (Path | str): filepath to store

        Returns:
            Tuple[zarr.storage.LocalStore, zarr.Group]: tuple containing the opened store and its root group
        """
        if file not in self.stores:
            store = zarr.storage.LocalStore(file, read_only=True)
            rootgroup = zarr.open_group(store.root)
            self.stores[file] = (store, rootgroup)

        return self.stores[file]

    def close(self) -> None:
        "Close all open zarr stores."
        for store, _ in self.stores.values():
            store.close()
        self.stores.clear()

    def __del__(self):
        "Cleanup on deletion."
        self.close()

    def map_index(self, idx: int) -> Tuple[str | Path, int]:
        """Map a global index to a specific file and local index within that file.

        Args:
            idx (int): The global index to map.

        Raises:
            RuntimeError: If the index cannot be mapped to any file.

        Returns:
            Tuple[str | Path, int]: The file and local index corresponding to the global index.
        """
        original_index = idx
        final_file: Path | str | None = None

        for size, dfile in zip(self._num_samples_per_file, self.input):
            if idx < size:
                final_file = dfile
                break
            else:
                idx -= size

        if final_file is None:
            raise RuntimeError(
                f"Error, index {original_index} could not be found in the supplied data files of size {self._num_samples_per_file} with total size {self._num_samples}"
            )
        return final_file, idx

    def get(self, idx: int) -> Data:
        """Get a single data sample by index."""
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        # Load the data from the processed files
        if self.preprocess:
            datapoint = torch.load(
                Path(self.processed_dir) / f"data_{idx}.pt", weights_only=False
            )
            if self.transform is not None:
                datapoint = self.transform(datapoint)
        else:
            # TODO: this is inefficient, but it's the only robust way I could find
            dfile, idx = self.map_index(idx)
            _, rootgroup = self._get_store_group(dfile)
            datapoint = self.data_reader(
                rootgroup,
                idx,
                self.float_type,
                self.int_type,
                self.validate_data,
            )

        return datapoint

    def __getitem__(
        self, idx: int | Sequence[int]
    ) -> Data | Sequence[Data] | Collection[Any]:
        """_summary_

        Args:
            idx (int | Sequence[int]): _description_

        Returns:
            Data | Sequence[Data] | Collection[Any]: _description_
        """
        if isinstance(idx, int):
            return self.get(idx)
        else:
            return [self.get(i) for i in idx]

    def len(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.preprocess:
            return len(self.processed_file_names)
        else:
            return self._num_samples
