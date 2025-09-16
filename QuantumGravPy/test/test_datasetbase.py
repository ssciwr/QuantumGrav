import pytest
import QuantumGrav as QG
from pathlib import Path
import torch
from torch_geometric.data import Data
import h5py
import zarr


def test_dataset_base_creation(create_data_hdf5, tmp_path):
    _, datafiles = create_data_hdf5

    dataset = QG.dataset_base.QGDatasetBase(
        input=datafiles,
        output=tmp_path,
        mode="hdf5",
        reader=lambda file: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=4,
        chunksize=300,
    )

    assert dataset.input == datafiles
    assert dataset.output == tmp_path
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


def test_dataset_base_creation_fails_bad_datafile(create_data_hdf5, tmp_path):
    _, datafiles = create_data_hdf5

    with pytest.raises(
        FileNotFoundError, match="Input file nonexistent_path does not exist."
    ):
        QG.dataset_base.QGDatasetBase(
            input=datafiles
            + [
                "nonexistent_path",
            ],
            output=tmp_path,
            mode="hdf5",
            reader=lambda file: [],
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_base_creation_fails_no_data_reader(create_data_hdf5, tmp_path):
    _, datafiles = create_data_hdf5

    with pytest.raises(
        ValueError, match="A reader function must be provided to load the data."
    ):
        QG.dataset_base.QGDatasetBase(
            input=datafiles,
            output=tmp_path,
            mode="hdf5",
            reader=None,
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_dataset_base_process_chunk_hdf5(create_data_hdf5, read_data, n):
    datadir, datafiles = create_data_hdf5

    dataset = QG.dataset_base.QGDatasetBase(
        input=datafiles,
        output=datadir,
        reader=read_data,
        mode="hdf5",
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=4,
    )

    with h5py.File(datafiles[0], "r") as raw_file:
        results = dataset.process_chunk_hdf5(
            raw_file,
            0,
            pre_transform=lambda x: x,
            pre_filter=lambda x: True,
        )

    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)


@pytest.mark.parametrize("mode", ["fallback", "normal"], ids=["fallback", "normal"])
@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_dataset_base_process_chunk_zarr(
    read_data, create_data_zarr, create_data_zarr_basic, mode, n
):
    if mode == "fallback":
        datadir, datafiles = create_data_zarr_basic
    else:
        datadir, datafiles = create_data_zarr

    dataset = QG.dataset_base.QGDatasetBase(
        input=datafiles,
        output=datadir,
        reader=read_data,
        mode="zarr",
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=4,
    )
    with zarr.storage.LocalStore(datafiles[0], read_only=True) as raw_file:
        results = dataset.process_chunk_zarr(
            raw_file, 0, pre_transform=lambda x: x, pre_filter=lambda x: True
        )
    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)
