# pytorch and torch geometric imports
from torch_geometric.data import Data
import torch

# data handling
import h5py
import json

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from multiprocessing import Pool


class QGDatasetMixin:
    """Mixin class that provides common functionality for the dataset classes. Works only for file-based datasets. Provides methods for processing data."""

    def __init__(
        self,
        input: list[str | Path],
        get_metadata: Callable[[str | Path], dict] | None = None,
        reader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
    ):
        """Initialize a DatasetMixin instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets. It provides a common interface for both in-memory and on-disk datasets. It is not to be instantiated directly, but rather used as a mixin for other dataset classes.

        Args:
            input (list[str  |  Path] : The list of input files for the dataset, or a callable that generates a set of input files.
            get_metadata (Callable[[str  |  Path], dict] | None, optional): A function to retrieve metadata for the dataset. Defaults to None.
            reader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from a file. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point values. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integer values. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after loading. Defaults to True.

        Raises:
            ValueError: If one of the input data files is not a valid HDF5 file
            ValueError: If the metadata retrieval function is invalid.
            FileNotFoundError: If an input file does not exist.
        """
        if reader is None:
            raise ValueError("A reader function must be provided to load the data.")

        if get_metadata is None:
            raise ValueError("A metadata retrieval function must be provided.")

        self.input = input
        self.data_reader = reader
        self.get_metadata = get_metadata
        self.metadata = {}
        self.float_type = float_type
        self.int_type = int_type
        self.validate_data = validate_data

        self._num_samples = None

        # ensure the input is a list of paths
        if self.processed_dir is not None:
            with open(Path(self.processed_dir) / "metadata.json", "r") as f:
                self.metadata = json.load(f)

        # get the number of samples in the dataset
        self._num_samples = 0
        for file in self.input:
            if not Path(file).exists():
                raise FileNotFoundError(f"Input file {file} does not exist.")
            with h5py.File(file, "r") as f:
                self._num_samples += f["num_causal_sets"][()]

    @property
    def processed_dir(self) -> str | None:
        """Get the path to the processed directory.

        Returns:
            str: The path to the processed directory, or None if it doesn't exist.
        """
        processed_path = Path(self.root) / "processed"
        if not processed_path.exists():
            return None
        return str(processed_path)

    @property
    def processed_files(self) -> list[str]:
        """Get a list of processed files in the processed directory.

        Returns:
            list[str]: A list of processed file paths, excluding JSON files.
        """

        if not Path(self.processed_dir).exists():
            return []

        return [
            str(Path(self.processed_dir) / f)
            for f in Path(self.processed_dir).iterdir()
            if f.is_file() and f.suffix == ".pt"  # Only include .
        ]

    @property
    def raw_file_names(self) -> list[str]:
        """Get the raw file names from the input list.

        Returns:
            list[str]: A list of raw file names.
        """
        return [Path(f).name for f in self.input if Path(f).suffix == ".h5"]

    @property
    def processed_file_names(self) -> list[str]:
        """Get the processed file names from the processed directory.

        Returns:
            list[str]: A list of processed file names.
        """
        if not Path(self.root).exists():
            return []

        return [f for f in Path(self.root).iterdir() if f.suffix == ".pt"]

    def write_data(self, data: list[Data]) -> None:
        """Write the processed data to disk.

        Args:
            data (list[Data]): The list of Data objects to write to disk.
        """
        if not Path(self.processed_dir).exists():
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        for i, d in enumerate(data):
            if d is not None:
                file_path = Path(self.processed_dir) / f"data_{i}.pt"
                torch.save(d, file_path)

    def process_chunk(
        self,
        raw_file: h5py.File,
        start: int,
        chunksize: int,
        n_processes: int = 1,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> Data | None:
        """Process a chunk of data from the raw file.

        Args:
            raw_file (h5py.File): The raw HDF5 file to read from.
            start (int): The starting index of the chunk.
            chunksize (int): The size of the chunk.
            n_processes (int, optional): The number of processes to use for parallel processing. Defaults to 1.
            pre_transform (Callable[[Data], Data] | None, optional): Transformation that adds additional features to the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function that filters the data. Defaults to None.

        Returns:
            Data | None: The processed data or None if the chunk is empty.
        """

        # we can't rely on being able to read from the raw_files in parallel, so we need to read the data sequentially
        data = [
            self.data_reader(
                raw_file,
                i,
                self.float_type,
                self.int_type,
                self.validate_data,
            )
            for i in range(start, start + chunksize)
        ]

        results = []
        if n_processes > 1:

            def process_item(item):
                if pre_filter is not None and not pre_filter(item):
                    return None
                if pre_transform is not None:
                    return pre_transform(item)
                return item

            with Pool(n_processes) as pool:
                # Process items as they complete, in any order
                for result in pool.imap_unordered(process_item, data):
                    if result is not None:
                        results.append(result)
        else:
            for datapoint in data:
                if pre_filter is not None and not pre_filter(datapoint):
                    continue
                if pre_transform is not None:
                    datapoint = pre_transform(datapoint)
        return results
