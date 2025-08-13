import QuantumGrav as QG
import torch
from pathlib import Path
import pytest
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@pytest.mark.parametrize("n", [1, 3], ids=["sequential", "parallel"])
def test_ondisk_dataset_creation_processing(create_data_hdf5, read_data, n):
    datadir, datafiles = create_data_hdf5
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


def test_ondisk_dataset_with_dataloader(create_data, read_data):
    datadir, datafiles = create_data
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


def test_ondisk_dataset_get(create_data, read_data):
    datadir, datafiles = create_data
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

    sample = dataset.get(12)
    assert isinstance(sample, Data)

    with pytest.raises(IndexError, match="Index out of bounds."):
        dataset.get(20)
