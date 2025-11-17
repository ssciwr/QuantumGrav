# pytorch and torch geometric imports
from torch_geometric.data import Data
import torch

# data handling
import yaml
import zarr

# system imports and quality of life tools
from pathlib import Path
from collections.abc import Callable, Collection
from joblib import delayed, Parallel
from typing import Sequence


class QGDatasetBase:
    """Mixin class that provides common functionality for the dataset classes. Works only for file-based datasets. Provides methods for processing data."""

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        reader: Callable[
            [zarr.Group, torch.dtype, torch.dtype, bool],
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
            reader (Callable[[zarr.Group, torch.dtype, torch.dtype, bool], list[Data]] | None, optional): A function to load data from a file. Defaults to None.
            float_type (torch.dtype, optional): The data type to use for floating point values. Defaults to torch.float32.
            int_type (torch.dtype, optional): The data type to use for integer values. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate the data after loading. Defaults to True.
            n_processes (int, optional): The number of processes to use for parallel processing of read data. Defaults to 1.
            chunksize (int, optional): The size of the chunks to process in parallel. Defaults to 1000.

        Raises:
            ValueError: If one of the input data files does not exist
            ValueError: If the metadata retrieval function is invalid.
            FileNotFoundError: If an input file does not exist.
        """
        if reader is None:
            raise ValueError("A reader function must be provided to load the data.")

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
        num_samples_per_file = []
        for filepath in self.input:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Input file {filepath} does not exist.")
            n = self._get_num_samples_per_file(filepath)
            num_samples_per_file.append(n)
            self._num_samples += n

        self._num_samples_per_file = torch.tensor(num_samples_per_file)

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

    def _get_num_samples_per_file(self, filepath: str | Path) -> int:
        """Get the number of samples in a given file.

        Args:
            filepath (str | Path): The path to the file.

        Raises:
            ValueError: If the file is not a valid Zarr file.

        Returns:
            int: The number of samples in the file.
        """

        # try to find the sample number from a dedicated dataset
        def try_find_numsamples(f):
            s = None
            for name in ["num_causal_sets", "num_samples"]:
                if name in f:
                    s = f[name]
                    break
            return s

        # ... if that fails, we try to read it from any scalar dataset.
        # ... if we can´t because they are of unequal sizes, we return None
        # ... to indicate an unresolvable state
        def fallback(f) -> int | None:
            # find scalar datasets and use their sizes to determine size
            shapes = [f[k].shape[0] for k in f.keys() if len(f[k].shape) == 1]
            max_shape = max(shapes)
            min_shape = min(shapes)
            if max_shape != min_shape:
                return None
            else:
                return max_shape

        # same logic for Zarr
        try:
            group = zarr.open_group(
                zarr.storage.LocalStore(filepath, read_only=True),
                path="",
                mode="r",
            )
            # note that fallback returns an int directly,
            # while for try_find_numsamples we need to index into the result
            s = try_find_numsamples(group)
            if s is not None:
                return s[0]
            else:
                s = fallback(group)
                if s is not None:
                    return s
                else:
                    raise RuntimeError("Unable to determine number of samples.")
        except Exception:
            # we need an extra fallback for zarr b/c Julia Zarr and python Zarr
            # can differ in layout - Julia Zarr does not have to have a group
            try:
                store = zarr.storage.LocalStore(filepath, read_only=True)
                arr = zarr.open_array(store, path="adjacency_matrix")
                s = max(arr.shape)
                return s
            except Exception:
                raise

    @property
    def processed_dir(self) -> str:
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
        suf = ".zarr"
        return [str(Path(f).name) for f in self.input if Path(f).suffix == suf]

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

    def process_chunk(
        self,
        store: zarr.storage.LocalStore,
        start: int,
        pre_transform: Callable[[Data | Collection], Data] | None = None,
        pre_filter: Callable[[Data | Collection], bool] | None = None,
    ) -> Sequence[Data]:
        """Process a chunk of data from the raw file. This method is intended to be used in the data loading pipeline to read a chunk of data, apply transformations, and filter the read data, and thus should not be called directly.

        Args:
            store (zarr.storage.LocalStore): local zarr storage
            start (int): start index
            pre_transform (Callable[[Data], Data] | None, optional): Transformation that adds additional features to the data. Defaults to None.
            pre_filter (Callable[[Data], bool] | None, optional): A function that filters the data. Defaults to None.

        Returns:
            list[Data]: The processed data or None if the chunk is empty.
        """
        N = self._get_num_samples_per_file(store.root)
        rootgroup = zarr.open_group(store.root)

        def process_item(i: int):
            item = self.data_reader(
                rootgroup,
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
