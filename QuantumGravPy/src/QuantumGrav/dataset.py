# pytorch and torch geometric imports
from torch_geometric.data import Data, InMemoryDataset
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


class QuantumGravDatasetBase:
    pass


class QuantumGravDatasetInMemory(InMemoryDataset):
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
        writer: Callable[[Data, str], None] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        parallel_processing: bool = False,
        writer_kwargs: dict[str, Any] = None,
    ):
        self.writer_kwargs = writer_kwargs or {}

        if self.data_writer is None:
            raise ValueError("A writer function must be provided to write the data.")

        if loader is None:
            raise ValueError("A loader function must be provided to load the data.")

        if get_metadata is None:
            raise ValueError("A metadata retrieval function must be provided.")

        super().__init__(output, transform, pre_transform, pre_filter)
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
                with h5py.File(
                    file, "r"
                ) as f:  # TODO: memory map this perhaps for speed
                    read = 0
                    # read the data in chunks and process it parallelized or
                    # sequentially based on the parallel_processing flag
                    while read < f["num_causal_sets"][()]:
                        data_list = self.data_loader(
                            f,
                            float_type=self.float_type,
                            int_type=self.int_type,
                            validate=self.validate_data,
                        )

                        if self.parallel_processing:
                            processed = Parallel(n_jobs=-1)(
                                delayed(self.pre_transform)(data_point)
                                for data_point in data_list
                                if delayed(self.pre_filter)(data_point)
                            )
                        else:
                            processed = [
                                self.pre_transform(data_point)
                                for data_point in data_list
                                if self.pre_filter(data_point)
                            ]

                        for i, data_point in enumerate(processed):
                            self.data_writer(
                                data_point,
                                os.path.join(
                                    self.processed_dir,
                                    f"{self._num_samples + read + i}",
                                ),
                                self.writer_kwargs,
                            )

                        read += len(data_list)
