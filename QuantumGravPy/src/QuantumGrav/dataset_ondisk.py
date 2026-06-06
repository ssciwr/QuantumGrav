# pytorch and torch geometric imports
from torch_geometric.data import Data, Dataset
import torch

# data handling
import zarr

# system imports and quality of life tools
from pathlib import Path

from collections.abc import Callable, Sequence, Collection
from typing import Any, Tuple
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

# internals
from .utils import ZarrStore, identity, tautology
from .load_zarr import zarr_group_to_dict


def _process_data(
    file: Path | str,
    pre_filter: Callable,
    pre_transform: Callable,
    data_chunk: list[str],
    reader: Callable | None = None,
):
    out = []
    with ZarrStore(file, mode="r") as store:
        root = zarr.open_group(store, path="", mode="r")
        if reader is not None:
            for path in data_chunk:
                data = reader(root, path)
                if pre_filter(data):
                    out.append(pre_transform(data))
        else:
            for path in data_chunk:
                data = dict()
                zarr_group_to_dict(root[path], data)
                if pre_filter(data):
                    out.append(pre_transform(data))
    return out


class QGDataset(Dataset):
    """On-disk QuantumGrav dataset backed by zarr stores."""

    schema = {
        "type": "object",
        "description": "QGDataset constructor parameters expressed as a config dict",
        "properties": {
            "files": {
                "type": "array",
                "description": "List of zarr store paths to read from",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "output": {
                "type": "string",
                "description": "Directory where processed data will be stored",
            },
            "float_type": {
                "description": "Torch floating-point dtype for tensor conversion",
            },
            "int_type": {
                "description": "Torch integer dtype for tensor conversion",
            },
            "validate_data": {
                "type": "boolean",
                "description": "Whether to validate transformed data objects",
            },
            "n_processes": {
                "type": "integer",
                "minimum": 0,
                "description": "Worker processes to use for preprocessing",
            },
            "chunksize": {
                "type": "integer",
                "description": "Number of samples to process per chunk",
            },
            "transform": {
                "description": "Callable applied to each sample at read time",
            },
            "pre_transform": {
                "description": "Callable applied once before saving processed data",
            },
            "pre_filter": {
                "description": "Callable used to drop samples before pre_transform",
            },
            "reader": {
                "description": (
                    "reader(root, path) replaces the default zarr_group_to_dict. "
                    "root is the zarr root group, path is the group path string."
                ),
            },
        },
        "required": ["files", "output"],
    }

    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        chunksize: int = 1000,
        n_processes: int = 1,
        # dataset properties
        transform: Callable[..., Data] | None = None,
        pre_transform: Callable[..., Data] | None = None,
        pre_filter: Callable[..., bool] | None = None,
        reader: Callable | None = None,
    ):
        """Create a new QGDataset instance.

        Loads QuantumGrav data from zarr stores on disk. When neither ``pre_transform``
        nor ``pre_filter`` is given the dataset reads samples lazily at access time
        and no ``processed`` directory is written.

        Args:
            input (list[str | Path]): List of input zarr file paths.
            output (str | Path): Directory where processed data will be stored.
            float_type (torch.dtype, optional): Dtype for float tensors. Defaults to torch.float32.
            int_type (torch.dtype, optional): Dtype for int tensors. Defaults to torch.int64.
            validate_data (bool, optional): Whether to validate transformed data objects. Defaults to True.
            chunksize (int, optional): Number of samples processed per chunk. Defaults to 1000.
            n_processes (int, optional): Worker processes for preprocessing. Defaults to 1.
            transform (Callable | None, optional): Applied to each sample at read time. Defaults to None.
            pre_transform (Callable | None, optional): Applied once before saving processed data. Defaults to None.
            pre_filter (Callable | None, optional): Called before pre_transform; samples that return False are dropped. Defaults to None.
            reader (Callable | None, optional): ``reader(root, path)`` replaces the default
                ``zarr_group_to_dict`` when reading raw samples. ``root`` is the zarr root
                group of a store and ``path`` is the group path string (e.g. ``"cset_1"``).
                The return value is passed to ``pre_filter`` and ``pre_transform`` unchanged.
                Defaults to None.
        """
        does_preprocessing = pre_transform is not None or pre_filter is not None

        if pre_transform is None:
            pre_transform = identity

        if pre_filter is None:
            pre_filter = tautology

        if transform is None:
            transform = identity

        self.stores = {}

        self.input = [Path(filepath).resolve() for filepath in input]

        for filepath in self.input:
            if (Path(filepath).resolve()).exists() is False:
                raise FileNotFoundError(f"Input file {filepath} does not exist.")

        self.output = Path(output).resolve()
        self.metadata = {}
        self.float_type = float_type
        self.int_type = int_type
        self.validate_data = validate_data
        self.n_processes = n_processes
        self.chunksize = chunksize
        self.does_preprocessing = does_preprocessing
        self.reader = reader

        # ensure the input is a list of paths
        if Path(self.processed_dir).exists():
            with open(Path(self.processed_dir) / "metadata.yaml", "r") as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)

            self._num_samples = self.metadata["num_samples"]
            self._num_samples_per_file = self.metadata["num_samples_per_file"]

        else:
            # get the number of samples in the dataset
            self._num_samples = 0
            self._num_samples_per_file = {}

            for filepath in self.input:
                if not Path(filepath).exists():
                    raise FileNotFoundError(f"Input file {filepath} does not exist.")

                with ZarrStore(filepath, mode="r") as store:
                    root = zarr.open_group(
                        store,
                        path="",
                        mode="r",
                    )
                    n = len(root)
                    self._num_samples_per_file[str(filepath)] = n
                self._num_samples += n

            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            self.metadata = {
                "files": [str(Path(f).resolve().absolute()) for f in self.input],
                "num_samples_per_file": self._num_samples_per_file,
                "num_samples": int(self._num_samples),
                "input": [str(Path(f).resolve().absolute()) for f in self.input],
                "output": str(Path(self.output).resolve().absolute()),
                "float_type": str(self.float_type),
                "int_type": str(self.int_type),
                "validate_data": self.validate_data,
                "n_processes": self.n_processes,
                "chunksize": self.chunksize,
                "does_preprocessing": self.does_preprocessing,
            }

            with open(Path(self.processed_dir) / "metadata.yaml", "w") as f:
                yaml.dump(self.metadata, f)

        Dataset.__init__(
            self,
            root=output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def processed_dir(self) -> str:
        """Get the path to the processed directory.

        Returns:
            str: The path to the processed directory, or None if it doesn't exist.
        """
        processed_path = Path(self.output).resolve() / "processed"
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

    def process(self) -> None:
        """Process the dataset from the read rawdata into its final form."""
        if self.pre_filter is None and self.pre_transform is None:
            return

        def chunks(xs, n):
            for i in range(0, len(xs), n):
                yield xs[i : i + n]

        out_idx = 0

        for filepath in tqdm(self.input, desc="zarr stores"):
            datapoints = self._num_samples_per_file[str(filepath)]

            datagroups = list(
                chunks([f"cset_{i + 1}" for i in range(datapoints)], self.chunksize)
            )

            n_datagroups = len(datagroups)

            reading_groups = chunks(
                [
                    (filepath, self.pre_filter, self.pre_transform, chunk, self.reader)
                    for chunk in datagroups
                ],
                10 if n_datagroups > 10 else 1,
            )

            parallel = Parallel(
                n_jobs=self.n_processes,
            )
            for chunk in reading_groups:
                processed = parallel(delayed(_process_data)(*args) for args in chunk)

                for datalist in processed:
                    for data in datalist:
                        if data is not None:
                            torch.save(
                                data, Path(self.processed_dir) / f"data_{out_idx}.pt"
                            )
                            out_idx += 1

    def _get_store_group(
        self, file: Path | str
    ) -> Tuple[zarr.storage.LocalStore | zarr.storage.ZipStore, zarr.Group]:
        """Get a requested open store and add it to an internal cache if not open yet.

        Args:
            file (Path | str): filepath to store

        Returns:
            Tuple[zarr.storage.LocalStore, zarr.Group]: tuple containing the opened store and its root group
        """
        if file not in self.stores:
            if Path(file).suffix == ".zip":
                store = zarr.storage.ZipStore(file, mode="r")
            else:
                store = zarr.storage.LocalStore(file, read_only=True)

            rootgroup = zarr.open_group(
                store,
                path="",
                mode="r",
            )
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

        for dfile, size in self._num_samples_per_file.items():
            if idx < size:
                final_file = dfile
                break
            else:
                idx -= size

        if final_file is None:
            raise RuntimeError(
                f"Error, index {original_index} could not be found in the supplied data files of size {list(self._num_samples_per_file.values())} with total size {self._num_samples}"
            )
        return final_file, idx

    def get(self, idx: int) -> Data:
        """Get a single data sample by index."""

        # Load the data from the processed files
        if self.does_preprocessing:
            datapoint = torch.load(
                Path(self.processed_dir) / f"data_{idx}.pt", weights_only=False
            )
        else:
            # TODO: this is inefficient, but it's the only robust way I could find
            dfile, idx = self.map_index(idx)
            store, root = self._get_store_group(dfile)
            # since julia Zarr files allow having no root groups, we need to open the store directly
            datapoint = dict()
            zarr_group_to_dict(
                zarr.open_group(
                    store,
                    path=f"cset_{idx + 1}",
                    mode="r",
                ),
                datapoint,
            )
        datapoint = self.transform(datapoint)

        return datapoint

    def __getitem__(
        self, idx: int | Sequence[int] | slice
    ) -> Data | Sequence[Data] | Collection[Any]:
        if isinstance(idx, int):
            return self.get(idx)
        elif isinstance(idx, slice):
            return [self.get(i) for i in range(idx.start, idx.stop, idx.step or 1)]
        else:
            return [self.get(i) for i in idx]

    def len(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.does_preprocessing:
            return len(self.processed_file_names)
        else:
            return self._num_samples
