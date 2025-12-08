import QuantumGrav as QG
from pathlib import Path
import re
import pytest
import zarr

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@pytest.mark.parametrize("n", [1, 3], ids=["sequential", "parallel"])
def test_ondisk_dataset_creation_processing(request, create_data_zarr, read_data, n):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=4,
        transform=lambda x: x,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    assert dataset.input == datafiles
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.output == datadir
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.data_reader is not None
    assert dataset.chunksize == 4
    assert dataset.n_processes == n
    assert len(dataset) == 15  # Assuming 15 samples in the datafiles
    assert Path(dataset.processed_dir).exists()
    assert (Path(dataset.processed_dir) / "metadata.yaml").exists()
    assert all(f"data_{i}.pt" in dataset.processed_file_names for i in range(15))
    assert isinstance(dataset[5], Data)


def test_ondisk_dataset_creation_processing_no_pre_transform(
    create_data_zarr, read_data_dict
):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data_dict,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
        transform=lambda x: x,
    )

    assert dataset.input == datafiles
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.output == datadir
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.data_reader is not None
    assert dataset.chunksize == 4
    assert dataset.n_processes == 1
    assert len(dataset) == 15  # Assuming 15 samples in the datafiles
    assert len(dataset.stores) == 0
    assert Path(dataset.processed_dir).exists() is True  # always will exist
    assert isinstance(dataset[5], dict)


def test_ondisk_dataset_map_index(create_data_zarr, read_data_dict):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data_dict,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=2,
        chunksize=4,
        transform=lambda x: x,
    )

    assert len(dataset.input) == 3
    assert all(dataset._num_samples_per_file == [5, 5, 5])
    assert dataset._num_samples == 15
    assert dataset.map_index(3) == (datafiles[0], 3)
    assert dataset.map_index(12) == (datafiles[2], 2)

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Error, index 15 could not be found in the supplied data files of size [5 5 5] with total size 15"
        ),
    ):
        dataset.map_index(15)


def test_ondisk_dataset_get(create_data_zarr, read_data_dict):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data_dict,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=2,
        chunksize=4,
        transform=lambda x: x,
    )
    assert len(dataset.stores) == 0

    _ = dataset[0]
    assert len(dataset.stores) == 1

    assert len(dataset.stores[dataset.input[0]]) == 2
    assert isinstance(dataset.stores[dataset.input[0]][0], zarr.storage.LocalStore)
    assert isinstance(dataset.stores[dataset.input[0]][1], zarr.Group)

    _ = dataset[3]
    assert len(dataset.stores) == 1

    _ = (dataset[6],)
    assert len(dataset.stores) == 2

    _ = dataset[12]
    assert len(dataset.stores) == 3

    for file in dataset.input:
        assert file in dataset.stores
    dataset.close()
    assert len(dataset.stores) == 0


def test_ondisk_dataset_with_dataloader(create_data_zarr, read_data):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=3,
        transform=lambda x: x,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )
    assert len(loader) == 8  # Assuming 15 samples and batch size of 2

    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert len(batch) == 2 if i < 7 else 1  # Last batch may be smaller
