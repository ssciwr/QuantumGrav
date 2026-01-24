import pytest
import zarr
import numpy as np
from typing import Callable, Type, Dict, Any
import shutil

import QuantumGrav as QG


import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader


# data fixtures
@pytest.fixture(scope="session")
def create_data_zarr_basic(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("test_data_quantumgrav", numbered=True)

    datafiles = []

    for i in range(3):
        data = []
        for _ in range(5):
            num_nodes = np.random.randint(10, 15)
            adjacency_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            link_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            max_pathlen_future = np.random.rand(num_nodes).astype("float32")
            max_pathlen_past = np.random.rand(num_nodes).astype("float32")
            dimension = np.random.randint(2, 10)
            atomcount = num_nodes

            data.append(
                {
                    "adjacency_matrix": adjacency_matrix,
                    "link_matrix": link_matrix,
                    "max_pathlen_future": max_pathlen_future,
                    "max_pathlen_past": max_pathlen_past,
                    "dimension": dimension,
                    "atomcount": atomcount,
                }
            )

        # Save the data to an zarr file
        zarr_file = tmpdir / f"test_data_{i}.zarr"

        if zarr_file.exists():
            shutil.rmtree(zarr_file)

        store = zarr.storage.LocalStore(zarr_file, read_only=False)

        adj = zarr.create_array(
            store,
            shape=(len(data), 15, 15),
            chunks=(1, 15, 15),
            dtype="float32",
            name="adjacency_matrix",
        )

        link = zarr.create_array(
            store,
            shape=(len(data), 15, 15),
            chunks=(1, 15, 15),
            dtype="float32",
            name="link_matrix",
        )

        maxpathlen_future = zarr.create_array(
            store,
            shape=(len(data), 15),
            chunks=(1, 15),
            name="max_pathlen_future",
            dtype="float32",
        )

        max_pathlen_past = zarr.create_array(
            store,
            shape=(len(data), 15),
            chunks=(1, 15),
            name="max_pathlen_past",
            dtype="float32",
        )

        dimension = zarr.create_array(
            store, shape=(len(data)), chunks=(1,), name="dimension", dtype="int32"
        )

        atomcount = zarr.create_array(
            store, shape=(len(data)), chunks=(1,), name="atomcount", dtype="int32"
        )

        for j, d in enumerate(data):
            adjmat = d["adjacency_matrix"]
            linkmat = d["link_matrix"]
            max_path_f = d["max_pathlen_future"]
            max_path_p = d["max_pathlen_past"]
            adj[j, 0 : adjmat.shape[0], 0 : adjmat.shape[1]] = adjmat
            link[j, 0 : linkmat.shape[0], 0 : linkmat.shape[1]] = linkmat
            maxpathlen_future[j, 0 : max_path_f.shape[0]] = max_path_f
            max_pathlen_past[j, 0 : max_path_p.shape[0]] = max_path_p
            dimension[j] = d["dimension"]
            atomcount[j] = d["atomcount"]

        datafiles.append(zarr_file)

    yield tmpdir, datafiles

    # remove created files again
    for file in datafiles:
        if file.exists():
            shutil.rmtree(file)
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def create_data_zarr(create_data_zarr_basic):
    tmpdir, datafiles = create_data_zarr_basic

    for file in datafiles:
        # Save the data to an zarr file
        store = zarr.storage.LocalStore(file, read_only=False)

        dims = zarr.open_array(store, path="dimension")
        if "num_samples" in store.root.iterdir() is False:
            num_samples = zarr.create_array(
                store, shape=(1,), chunks=(1,), name="num_samples", dtype="int32"
            )
            num_samples[0] = dims.shape[0]

    yield tmpdir, datafiles


@pytest.fixture
def read_data_dict():
    def reader(
        store, idx: int, float_dtype: Type, int_dtype: Type, validate: Callable
    ) -> Dict[Any, Any]:
        root = zarr.open_group(store, mode="r")

        adj_raw = root["adjacency_matrix"][idx, :, :]
        adj_matrix = torch.tensor(adj_raw, dtype=float_dtype)

        # Path lengths
        max_path_future = torch.tensor(
            root["max_pathlen_future"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        max_path_past = torch.tensor(
            root["max_pathlen_past"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        return {
            "adj": adj_matrix,
            "max_path_future": max_path_future,
            "max_path_past": max_path_past,
            "dimension": root["dimension"][idx],
        }

    return reader


@pytest.fixture
def read_data(read_data_dict):
    def reader(f: zarr.Group, idx: int, float_dtype, int_dtype, validate) -> Data:
        datadict = read_data_dict(f, idx, float_dtype, int_dtype, validate)

        adj_matrix = datadict["adj"].detach().clone()
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        node_features = []
        node_features.extend([datadict["max_path_future"], datadict["max_path_past"]])

        x = torch.cat(node_features, dim=1)

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
                dtype=int_dtype,
            ),
        )

        if validate and not data.validate():
            raise ValueError("Data validation failed.")
        return data

    return reader


@pytest.fixture
def make_dataset(create_data_zarr, read_data):
    datadir, datafiles = create_data_zarr

    dataset = QG.QGDataset(
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
    return dataset


@pytest.fixture
def make_dataloader(create_data_zarr, make_dataset):
    _, __ = create_data_zarr

    dataset = make_dataset
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,  # Ensure all batches are of the same size. last batches that are bad need to be handled by hand
    )
    return dataloader


@pytest.fixture(scope="session")
def yaml_text():
    yaml_text = """
        name: test_model
        model:
            layers: !sweep
                values: [1, 2]

            type: !pyobject QuantumGrav.models.GNNBlock
            convtype: !pyobject torch_geometric.nn.SAGEConv
            bs: !coupled-sweep
                target: model.layers
                values: [16, 32]
            lr: !sweep
                values: [0.1, 0.01, 0.001]
            foo:
                -
                    x: 3
                    y: 5
                -
                    x: !sweep
                        values: [1, 2]
                    y: 2
            bar:
                - x: !coupled-sweep
                    target: model.foo[1].x
                    values: [-1, -2]
            baz:
                - x: !coupled-sweep
                    target: model.foo.1.x
                    values: [-10, -20]

            listsweep: !sweep
                values:
                    - [1, 2]
                    - [3, 4, 5]
                    - [6]

            coupled_listsweep: !coupled-sweep
                target: model.listsweep
                values:
                    - [10, 20]
                    - [30, 40, 50]
                    - [60]

        trainer:
            epochs: !range
                start: 1
                stop: 6
                step: 2

            lr: !random_uniform
                start: 1e-5
                stop: 1e-2
                log: true
                size: 4

            lr_2: !random_uniform
                start: 0.1
                stop: 1.0
                log: false

            drop_rate: !range
                start: 0.1
                stop: 0.5
                step: 0.2

            foo_ref: !reference
                target: model.foo[1].x
        """
    return yaml_text
