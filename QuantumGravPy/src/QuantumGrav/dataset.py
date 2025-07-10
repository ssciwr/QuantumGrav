# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch


# data handling
import h5py
import json

# system imports and quality of life tools
from pathlib import Path
import os
from collections.abc import Callable
from typing import Any
from joblib import Parallel, delayed


class QuantumGravDatasetMixin:
    def __init__(
        self,
        input: list[str | Path] | Callable[[Any], dict],
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        self.writer_kwargs = writer_kwargs or {}
        if loader is None:
            raise ValueError("A loader function must be provided to load the data.")

        if get_metadata is None:
            raise ValueError("A metadata retrieval function must be provided.")

        self.input = input
        self._num_samples = None
        self.data_loader = loader
        self.data_writer = writer
        self.get_metadata = get_metadata
        self.metadata = {}
        self.float_type = float_type
        self.int_type = int_type
        self.validate_data = validate_data
        self.parallel_processing = parallel_processing

        # ensure the input is a list of paths
        if self.processed_dir is not None:
            with open(os.path.join(self.processed_dir, "metadata.json"), "r") as f:
                self.metadata = json.load(f)

        # get the number of samples in the dataset
        self._num_samples = 0
        for file in self.input:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist.")
            with h5py.File(file, "r") as f:
                self._num_samples += f["num_causal_sets"][()]

    @property
    def processed_dir(self) -> str | None:
        """_summary_

        Returns:
            str: _description_
        """
        processed_path = os.path.join(self.root, "processed")
        if not os.path.exists(processed_path):
            return None
        return processed_path

    @property
    def processed_files(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """

        if not os.path.isdir(self.processed_dir):
            return []

        return [
            os.path.join(self.processed_dir, f)
            for f in os.listdir(self.processed_dir)
            if f.endswith(".json") is False
        ]

    @property
    def raw_file_names(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        return [os.path.basename(f) for f in self.input if Path(f).suffix == ".h5"]

    @property
    def processed_file_names(self) -> list[str]:
        """_summary_

        Returns:
            list[str]: _description_
        """
        if os.path.isdir(self.root) is False:
            return []

        return [f for f in os.listdir(self.root) if f.endswith(".pt")]

    def process_chunk(
        self,
        raw_file: h5py.File,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> tuple[int, list[Data]]:
        """Process a chunk of data from the raw HDF5 file.

        Args:
            raw_file (h5py.File): The raw HDF5 file to process data from.

        Returns:
            list[Data]: A list of processed data items.
        """
        data_list = self.data_loader(
            raw_file,
            float_type=self.float_type,
            int_type=self.int_type,
            validate=self.validate_data,
        )

        if self.parallel_processing:
            processed = Parallel(n_jobs=-1)(
                delayed(pre_transform)(data_point)
                for data_point in data_list
                if delayed(pre_filter)(data_point)
            )
        else:
            processed = [
                pre_transform(data_point)
                for data_point in data_list
                if pre_filter(data_point)
            ]
        return len(data_list), processed


class QuantumGravDatasetInMemory(QuantumGravDatasetMixin, InMemoryDataset):
    """A dataset class for QuantumGrav data that can be loaded into memory."""

    def __init__(
        self,
        input: list[str | Path] | Callable[[Any], dict],
        output: str | Path,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        QuantumGravDatasetMixin.__init__(
            input,
            get_metadata,
            loader,
            writer,
            float_type,
            int_type,
            validate_data,
            parallel_processing,
            writer_kwargs,
        )

        InMemoryDataset.__init__(
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

            for file in self.raw_paths:
                with h5py.File(file, "r") as raw_file:
                    num_read = 0
                    # read the data in chunks and process it parallelized or
                    # sequentially based on the parallel_processing flag
                    while num_read < raw_file["num_causal_sets"][()]:
                        read_raw, processed = self.process_chunk(
                            raw_file,
                            pre_transform=self.pre_transform,
                            pre_filter=self.pre_filter,
                        )
                        if self.data_writer is None:
                            self.save(processed, self.processed_dir, self.writer_kwargs)
                        else:
                            self.data_writer(
                                processed,
                                self.processed_dir,
                                self.writer_kwargs,
                            )

                        num_read += read_raw


class QuantumGravDataset(QuantumGravDatasetMixin, Dataset):
    def __init__(
        self,
        input: list[str | Path] | Callable[[Any], dict],
        output: str | Path,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
        get_metadata: Callable[[str | Path], dict] | None = None,
        loader: Callable[[h5py.File, torch.dtype, torch.dtype, bool], list[Data]]
        | None = None,
        writer: Callable[[list[Data], str, dict[Any, Any]], None] | None = None,
        reader: Callable[[str | Path, int], Data] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        if writer is None:
            raise ValueError("A writer function must be provided to save the data.")

        if reader is None:
            raise ValueError("A reader function must be provided to read the data.")

        QuantumGravDatasetMixin.__init__(
            input,
            get_metadata,
            loader,
            writer,
            float_type,
            int_type,
            validate_data,
            parallel_processing,
            writer_kwargs,
        )

        Dataset.__init__(
            output,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self.reader = reader

        if self.processed_dir is not None:
            self.cached_processed_files = self.processed_files
        else:
            self.cached_processed_files = []

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

            for file in self.raw_paths:
                with h5py.File(file, "r") as raw_file:
                    num_read = 0
                    # read the data in chunks and process it parallelized or
                    # sequentially based on the parallel_processing flag
                    while num_read < raw_file["num_causal_sets"][()]:
                        read_raw, processed = self.process_chunk(
                            raw_file,
                            pre_transform=self.pre_transform,
                            pre_filter=self.pre_filter,
                        )

                        num_read += read_raw

                        # TODO: write out data

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_samples

    def get(self, idx: int) -> Data:
        """Get a single data point by index."""
        if self._num_samples is None:
            raise ValueError("Dataset has not been processed yet.")

        if idx < 0 or idx >= self._num_samples:
            raise IndexError("Index out of bounds.")

        # Load the data from the processed files
        data = self.reader(os.path.join(self.processed_dir), idx)
        if self.transform is not None:
            data = self.transform(data)
        return data
