# pytorch and torch geometric imports
from torch_geometric.data import Data
import torch

# data handling
import h5py
import yaml
import zarr

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable
from joblib import delayed, Parallel


class QGDatasetBase:
    """Mixin class that provides common functionality for the dataset classes. Works only for file-based datasets. Provides methods for processing data."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        mode: str = "hdf5",
        reader: Callable[
            [h5py.File | zarr.abc.store.Store, torch.dtype, torch.dtype, bool],
            list[Data],
        ]
        | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        n_processes: int = 1,
        chunksize: int = 1000,
        **kwargs,
    ):
        """Initialize a DatasetMixin instance. This class is designed to handle the loading, processing, and writing of QuantumGrav datasets. It provides a common interface for both in-memory and on-disk datasets. It is not to be instantiated directly, but rather used as a mixin for other dataset classes.

        Args:
            input (list[str  |  Path] : The list of input files for the dataset, or a callable that generates a set of input files.
            output (str | Path): The output directory where processed data will be stored.
            reader (Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from a file. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point values. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integer values. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after loading. Defaults to True.
            n_processes (int, optional): The number of processes to use for parallel processing of read data. Defaults to 1.
            chunksize (int, optional): The size of the chunks to process in parallel. Defaults to 1000.

        Raises:
            ValueError: If one of the input data files is not a valid HDF5 file
            ValueError: If the metadata retrieval function is invalid.
            FileNotFoundError: If an input file does not exist.
        """
        if reader is None:
            raise ValueError("A reader function must be provided to load the data.")

        if mode not in ["hdf5", "zarr"]:
            raise ValueError("mode must be 'hdf5' or 'zarr'")

        self.input = input
        for file in self.input:
            if Path(file).exists() is False:
                raise FileNotFoundError(f"Input file {file} does not exist.")

        self.output = output
        self.data_reader = reader
        self.metadata = {}
        self.float_type = float_type
        self.int_type = int_type
        self.validate_data = validate_data
        self.n_processes = n_processes
        self.chunksize = chunksize

        # get the number of samples in the dataset
        self._num_samples = 0

        for filepath in self.input:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Input file {filepath} does not exist.")

            if mode == "hdf5":
                with h5py.File(filepath, "r") as f:
                    self._num_samples += int(f["num_causal_sets"][()])
            else:
                with zarr.storage.LocalStore(filepath, read_only=True) as store:
                    root = zarr.open_group(store, path="", mode="r")
                    self._num_samples += int(root["num_samples"][0])

        # ensure the input is a list of paths
        if Path(self.processed_dir).exists():
            with open(Path(self.processed_dir) / "metadata.yaml", "r") as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            self.metadata = {
                "num_samples": int(self._num_samples),
                "input": [str(Path(f).resolve().absolute()) for f in self.input],
                "output": str(Path(self.output).resolve().absolute()),
                "float_type": str(self.float_type),
                "int_type": str(self.int_type),
                "validate_data": self.validate_data,
                "n_processes": self.n_processes,
                "chunksize": self.chunksize,
            }

            with open(Path(self.processed_dir) / "metadata.yaml", "w") as f:
                yaml.dump(self.metadata, f)

    @property
    def processed_dir(self) -> str | None:
        """Get the path to the processed directory.

        Returns:
            str: The path to the processed directory, or None if it doesn't exist.
        """
        processed_path = Path(self.output).resolve().absolute() / "processed"
        return str(processed_path)

    @property
    def raw_file_names(self) -> list[str]:
        """Get the raw file paths from the input list.

        Returns:
            list[str]: A list of raw file paths.
        """
        return [str(Path(f).name) for f in self.input if Path(f).suffix == ".h5"]

    @property
    def processed_file_names(self) -> list[str]:
        """Get a list of processed files in the processed directory.

        Returns:
            list[str]: A list of processed file paths, excluding JSON files.
        """

        if not Path(self.processed_dir).exists():
            return []
        return [
            str(f.name)
            for f in Path(self.processed_dir).iterdir()
            if f.is_file() and f.suffix == ".pt" and "data" in f.name
        ]

    def process_chunk_hdf5(
        self,
        raw_file: h5py.File,
        start: int,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> list[Data]:
        """Process a chunk of data from the raw file. This method is intended to be used in the data loading pipeline to read a chunk of data, apply transformations, and filter the read data, and thus should not be called directly.

        Args:
            raw_file (h5py.File): The raw HDF5 file to read from.
            start (int): The starting index of the chunk.
            pre_transform (Callable[[Data], Data] | None, optional): Transformation that adds additional features to the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function that filters the data. Defaults to None.

        Returns:
            list[Data]: The processed data or None if the chunk is empty.
        """

        # we can't rely on being able to read from the raw_files in parallel, so we need to read the data sequentially first
        data = [
            self.data_reader(
                raw_file,
                i,
                self.float_type,
                self.int_type,
                self.validate_data,
            )
            for i in range(
                start, min(start + self.chunksize, raw_file["num_causal_sets"][()])
            )
        ]

        def process_item(item):
            if pre_filter is not None and not pre_filter(item):
                return None
            if pre_transform is not None:
                return pre_transform(item)
            return item

        results = []
        if self.n_processes > 1:
            results = Parallel(n_jobs=self.n_processes)(
                delayed(process_item)(datapoint) for datapoint in data
            )
        else:
            results = [process_item(datapoint) for datapoint in data]
        return [res for res in results if res is not None]

    def process_chunk_zarr(
        self,
        store: zarr.storage.LocalStore,
        start: int,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> list[Data]:
        """Process a chunk of data from the raw file. This method is intended to be used in the data loading pipeline to read a chunk of data, apply transformations, and filter the read data, and thus should not be called directly.

        Args:
            store (zarr.storage.LocalStore): local zarr storage
            start (int): start index
            pre_transform (Callable[[Data], Data] | None, optional): Transformation that adds additional features to the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function that filters the data. Defaults to None.

        Returns:
            list[Data]: The processed data or None if the chunk is empty.
        """
        root = zarr.open_group(store, path="", mode="r")
        N = int(root["num_samples"][0])

        def process_item(i: int):
            item = self.data_reader(
                root,
                i,
                self.float_type,
                self.int_type,
                self.validate_data,
            )
            if pre_filter is not None and not pre_filter(item):
                return None
            if pre_transform is not None:
                return pre_transform(item)
            return item

        if self.n_processes > 1:
            results = Parallel(n_jobs=self.n_processes)(
                delayed(process_item)(i)
                for i in range(start, min(start + self.chunksize, N))
            )
        else:
            results = [
                process_item(i) for i in range(start, min(start + self.chunksize, N))
            ]

        return [res for res in results if res is not None]
