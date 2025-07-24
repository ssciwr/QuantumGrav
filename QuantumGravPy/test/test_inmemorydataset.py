import QuantumGrav as QG
import torch
import pytest
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


@pytest.mark.parametrize("n", [1, 3], ids=["sequential", "parallel"])
def test_inmemory_dataset_creation_and_process(create_data, read_data, n):
    datadir, datafiles = create_data

    dataset = QG.QGDatasetInMemory(
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
    assert "data.pt" in dataset.processed_file_names
    assert dataset.root == datadir
    assert dataset.output == datadir
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.chunksize == 4
    assert dataset.n_processes == n
    assert Path(dataset.processed_dir).exists()
    assert (Path(dataset.processed_dir) / "metadata.yaml").exists()
    assert len(dataset) == 15
    assert isinstance(dataset[5], Data)


def test_ondisk_dataset_with_dataloader(create_data, read_data):
    datadir, datafiles = create_data

    dataset = QG.QGDatasetInMemory(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
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
