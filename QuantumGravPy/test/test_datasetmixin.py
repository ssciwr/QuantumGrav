import pytest
import QuantumGrav as QG
from pathlib import Path
import torch
from typing import Callable
from torch_geometric.data import Data


def test_dataset_mixin_creation(create_data, tmp_path):
    _, datafiles = create_data

    dataset = QG.dataset_mixin.QGDatasetMixin(
        input=datafiles,
        output=tmp_path,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=lambda file: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=4,
        chunksize=300,
    )

    assert dataset.input == datafiles
    assert dataset.output == tmp_path
    assert isinstance(dataset.get_metadata, Callable)
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.data_reader is not None
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.chunksize == 300
    assert dataset.n_processes == 4
    assert Path(dataset.processed_dir).exists()
    assert (Path(dataset.processed_dir) / "metadata.yaml").exists()
    assert dataset.metadata["num_samples"] == 15
    assert dataset.metadata["input"] == [str(f.resolve().absolute()) for f in datafiles]
    assert dataset.metadata["output"] == str(Path(tmp_path).resolve().absolute())
    assert dataset.metadata["float_type"] == str(torch.float32)
    assert dataset.metadata["int_type"] == str(torch.int64)
    assert dataset.metadata["validate_data"] is True
    assert dataset.metadata["n_processes"] == 4
    assert dataset.metadata["chunksize"] == 300


def test_dataset_mixin_creation_fails_bad_datafile(create_data, tmp_path):
    _, datafiles = create_data
    with pytest.raises(
        FileNotFoundError, match="Input file nonexistent_path does not exist."
    ):
        QG.dataset_mixin.QGDatasetMixin(
            input=datafiles
            + [
                "nonexistent_path",
            ],
            output=tmp_path,
            get_metadata=lambda x: {"num_samples": len(x)},
            reader=lambda file: [],
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_mixin_creation_fails_no_metadata_reader(create_data, tmp_path):
    datadir, datafiles = create_data
    with pytest.raises(
        ValueError, match="A reader function must be provided to load the data."
    ):
        QG.dataset_mixin.QGDatasetMixin(
            input=datafiles,
            output=tmp_path,
            get_metadata=None,
            reader=None,
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_mixin_creation_fails_no_data_reader(create_data, tmp_path):
    _, datafiles = create_data
    with pytest.raises(
        ValueError, match="A reader function must be provided to load the data."
    ):
        QG.dataset_mixin.QGDatasetMixin(
            input=datafiles,
            output=tmp_path,
            get_metadata=lambda x: {"num_samples": len(x)},
            reader=None,
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_mixin_process_chunk_sequential(create_data, read_data):
    datadir, datafiles = create_data

    dataset = QG.dataset_mixin.QGDatasetMixin(
        input=datafiles,
        output=datadir,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
    )

    results = dataset.process_chunk(
        datafiles[0],
        0,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)


def test_dataset_mixin_process_chunk_parallel(create_data, read_data):
    datadir, datafiles = create_data

    dataset = QG.dataset_mixin.QGDatasetMixin(
        input=datafiles,
        output=datadir,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=2,
        chunksize=4,
    )

    results = dataset.process_chunk(
        datafiles[0],
        0,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)


def test_dataset_mixin_write_data(create_data, tmp_path, read_data):
    _, datafiles = create_data

    dataset = QG.dataset_mixin.QGDatasetMixin(
        input=datafiles,
        output=tmp_path,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
    )

    results = dataset.process_chunk(
        datafiles[0],
        0,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )
    dataset.write_data(results)
    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)
    assert Path(dataset.processed_dir).exists()
    assert len(dataset.processed_file_names) == 4

    assert all(
        f in dataset.processed_file_names for f in [f"data_{i}.pt" for i in range(4)]
    )

    new_dataset = QG.dataset_mixin.QGDatasetMixin(
        input=datafiles,
        output=tmp_path,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
    )

    assert new_dataset.processed_file_names == dataset.processed_file_names
    assert new_dataset.metadata == dataset.metadata
