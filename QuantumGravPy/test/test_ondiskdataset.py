import QuantumGrav as QG
from pathlib import Path
import re
import pytest
import zarr
from zarr.storage import ZipStore
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse


def pre_transform(datadict: dict) -> Data:
    adj_matrix = torch.tensor(datadict["adjacency_matrix"])
    edge_index, edge_weight = dense_to_sparse(adj_matrix)
    max_pathlen_future = torch.tensor(datadict["max_pathlen_future"]).unsqueeze(1)
    max_pathlen_past = torch.tensor(datadict["max_pathlen_past"]).unsqueeze(1)

    x = torch.cat((max_pathlen_past, max_pathlen_future), dim=1)

    dimension = datadict["dimension"]

    if isinstance(dimension, np.ndarray):
        value_list = [
            dimension.item(),
        ]
    else:
        value_list = [
            dimension,
        ]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1),
        y=torch.tensor(
            [
                value_list,
            ],
            dtype=torch.int32,
        ),
    )

    if not data.validate():
        raise ValueError("Data validation failed.")
    return data


@pytest.mark.parametrize("n", [1, 3], ids=["sequential", "parallel"])
def test_ondisk_dataset_creation_processing(create_data_zarr, n):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=10,
        transform=QG.utils.identity,
        pre_transform=pre_transform,
        pre_filter=QG.utils.tautology,
    )

    assert dataset.input == datafiles
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.output == datadir
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.chunksize == 10
    assert dataset.n_processes == n
    assert len(dataset) == 15  # Assuming 15 samples in the datafiles
    assert Path(dataset.processed_dir).exists()
    assert (Path(dataset.processed_dir) / "metadata.yaml").exists()
    assert all(f"data_{i}.pt" in dataset.processed_file_names for i in range(15))
    assert isinstance(dataset[5], Data)


def test_ondisk_dataset_creation_processing_no_pre_transform(create_data_zarr):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
        transform=QG.utils.identity,
    )

    assert dataset.input == datafiles
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.output == datadir
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.chunksize == 4
    assert dataset.n_processes == 1
    assert len(dataset) == 15  # Assuming 15 samples in the datafiles
    assert len(dataset.stores) == 0
    assert Path(dataset.processed_dir).exists() is True  # always will exist
    assert isinstance(dataset[5], dict)


def test_ondisk_dataset_map_index(create_data_zarr):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=2,
        chunksize=4,
        transform=lambda x: x,
    )

    assert len(dataset.input) == 3
    assert list(dataset._num_samples_per_file.values()) == [5, 5, 5]
    assert dataset._num_samples == 15
    assert dataset.map_index(3) == (str(datafiles[0]), 3)
    assert dataset.map_index(12) == (str(datafiles[2]), 2)

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Error, index 15 could not be found in the supplied data files of size [5, 5, 5] with total size 15"
        ),
    ):
        dataset.map_index(15)


def test_ondisk_dataset_get(create_data_zarr):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
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

    assert len(dataset.stores[str(dataset.input[0])]) == 2
    assert isinstance(dataset.stores[str(dataset.input[0])][0], zarr.storage.LocalStore)
    assert isinstance(dataset.stores[str(dataset.input[0])][1], zarr.Group)

    _ = dataset[3]
    assert len(dataset.stores) == 1

    _ = (dataset[6],)
    assert len(dataset.stores) == 2

    _ = dataset[12]
    assert len(dataset.stores) == 3

    _ = dataset[14]
    assert len(dataset.stores) == 3

    datarange = dataset[3:8]
    assert len(datarange) == 5

    datarange = dataset[[3, 4, 5, 6]]
    assert len(datarange) == 4

    for file in dataset.input:
        assert str(file) in dataset.stores
    dataset.close()
    assert len(dataset.stores) == 0


def test_ondisk_dataset_zip_store_get(create_data_zarr_zip, tmp_path):
    """Covers ZipStore branch in _get_store_group (preprocess=False reads live from zip)."""
    _, datafiles = create_data_zarr_zip
    dataset = QG.QGDataset(
        input=datafiles,
        output=tmp_path,
        float_type=torch.float32,
        int_type=torch.int64,
        n_processes=1,
        chunksize=4,
        transform=lambda x: x,
    )

    assert len(dataset) == 15
    _ = dataset[0]
    store, _ = dataset.stores[str(dataset.input[0])]
    assert isinstance(store, ZipStore)
    _ = dataset[6]  # second file
    assert len(dataset.stores) == 2
    assert isinstance(dataset[13], dict)
    dataset.close()
    assert len(dataset.stores) == 0


def test_ondisk_dataset_creation_processing_zip(create_data_zarr_zip, tmp_path):
    """Covers processing pipeline with zip inputs; uses n_processes=3 for actual parallel run."""
    _, datafiles = create_data_zarr_zip
    dataset = QG.QGDataset(
        input=datafiles,
        output=tmp_path,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=3,
        chunksize=5,
        transform=QG.utils.identity,
        pre_transform=pre_transform,
        pre_filter=QG.utils.tautology,
    )

    assert len(dataset) == 15
    assert all(f"data_{i}.pt" in dataset.processed_file_names for i in range(15))
    assert isinstance(dataset[0], Data)


def test_ondisk_dataset_file_not_found(tmp_path):
    """Covers the FileNotFoundError branch in __init__."""
    with pytest.raises(FileNotFoundError):
        QG.QGDataset(
            input=[tmp_path / "nonexistent.zarr"],
            output=tmp_path,
            float_type=torch.float32,
            int_type=torch.int64,
        )


def test_ondisk_dataset_getitem_list(create_data_zarr):
    """Covers the list-of-indices branch in __getitem__."""
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        n_processes=1,
        chunksize=4,
        transform=lambda x: x,
    )
    result = dataset[[0, 3, 7, 12]]
    assert len(result) == 4
    assert all(isinstance(item, dict) for item in result)


def test_ondisk_dataset_default_transform(create_data_zarr):
    """Covers the transform=None branch in __init__ (identity applied implicitly)."""
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        n_processes=1,
        chunksize=4,
    )
    assert isinstance(dataset[0], dict)


def test_ondisk_dataset_with_reader(create_data_zarr, tmp_path):
    """reader(root, path) replaces zarr_group_to_dict; output flows through pre_transform."""
    _, datafiles = create_data_zarr

    call_log = []

    def reader(root, path):
        call_log.append(path)
        grp = root[path]
        return {k: grp[k][:] for k in grp.array_keys()}

    dataset = QG.QGDataset(
        input=datafiles,
        output=tmp_path,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=5,
        pre_transform=pre_transform,
        reader=reader,
    )

    assert len(dataset) == 15
    item = dataset[0]
    assert isinstance(item, Data)
    assert item.x is not None
    assert item.x.shape == (15, 2)
    assert len(call_log) == 15


def test_ondisk_dataset_reader_returns_data_directly(create_data_zarr, tmp_path):
    """reader can return a Data object directly; pre_transform receives it unchanged."""
    _, datafiles = create_data_zarr

    def reader(root, path):
        return pre_transform({k: root[path][k][:] for k in root[path].array_keys()})

    dataset = QG.QGDataset(
        input=datafiles,
        output=tmp_path,
        float_type=torch.float32,
        int_type=torch.int64,
        n_processes=1,
        chunksize=5,
        pre_transform=lambda x: x,
        reader=reader,
    )

    assert len(dataset) == 15
    item = dataset[7]
    assert isinstance(item, Data)
    assert item.x is not None
    assert item.x.shape == (15, 2)


def test_ondisk_dataset_with_dataloader(create_data_zarr):
    datadir, datafiles = create_data_zarr
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
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
