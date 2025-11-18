import pytest
import juliacall as jcall
from pathlib import Path
import zarr
import numpy as np

import QuantumGrav as QG


import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader


# data
@pytest.fixture
def basic_transform():
    def transform(raw: jcall.DictValue) -> Data:
        # this function will transform the raw data dictionary from Julia into a PyTorch Geometric Data object. Hence, we have to deal with julia objects here
        adj_raw = raw["adjacency_matrix"]
        adj_matrix = torch.tensor(adj_raw, dtype=torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        adj_matrix = adj_matrix.to_sparse()

        node_features = []

        max_path_future = torch.tensor(
            raw["max_pathlen_future"], dtype=torch.float32
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        max_path_past = torch.tensor(
            raw["max_pathlen_past"], dtype=torch.float32
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        node_features.extend([max_path_future, max_path_past])

        # Concatenate all node features
        x = torch.cat(node_features, dim=1)
        classes = [
            raw["dimension"],
        ]
        y = torch.tensor(classes, dtype=torch.long)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
        )

        return data

    return transform


@pytest.fixture
def basic_converter():
    def converter(raw: jcall.DictValue) -> dict:
        # convert the raw data dictionary from Julia into a standard Python dictionary
        return {
            "dimension": int(raw["dimension"]),
            "atomcount": int(raw["atomcount"]),
            "adjacency_matrix": raw["adjacency_matrix"].to_numpy(),
            "link_matrix": raw["link_matrix"].to_numpy(),
            "max_pathlen_future": raw["max_pathlen_future"].to_numpy(),
            "max_pathlen_past": raw["max_pathlen_past"].to_numpy(),
        }

    return converter


@pytest.fixture
def jlcall_args():
    onthefly_config = {
        "seed": 42,
        "n_processes": 1,
        "batch_size": 5,
    }
    return onthefly_config


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object."""
    test_dir = Path(__file__).parent
    return test_dir.parent


@pytest.fixture(scope="session")
def test_dir(project_root):
    """Return the test directory."""
    return project_root / "test"


@pytest.fixture(scope="session")
def julia_paths(project_root, test_dir):
    """Return paths to Julia code and dependencies."""
    return {
        "jl_code_path": test_dir / "julia_testmodule.jl",
        "jl_base_module_path": project_root.parent / "QuantumGrav.jl",
        "jl_constructor_name": "Generator",
        "jl_dependencies": [
            "Distributions",
            "Random",
        ],
    }


@pytest.fixture
def jl_vars(julia_paths):
    return julia_paths


@pytest.fixture
def create_data(tmp_path_factory, julia_paths):
    datafiles = []
    tmpdir = tmp_path_factory.mktemp("test_data_quantumgrav")

    # make the julia module available
    path = str(Path(__file__).parent.joinpath("julia_testmodule.jl"))

    jl_module = jcall.newmodule("test_qg")

    for dep in julia_paths["jl_dependencies"]:
        jl_module.seval(f'using Pkg; Pkg.add("{dep}")')

    jl_module.seval(
        f'using Pkg; Pkg.develop(path="{julia_paths["jl_base_module_path"]}", name="QuantumGrav")'
    )  # only for now -> get from package index later
    jl_module.seval(f'include("{path}")')
    generator_constructor = getattr(jl_module, "Generator")
    jl_generator = generator_constructor(
        {
            "seed": 42,
        }
    )

    return path, datafiles, tmpdir, jl_generator


@pytest.fixture
def create_data_zarr_basic(create_data):
    path, datafiles, tmpdir, jl_generator = create_data

    for i in range(3):
        data = jl_generator(5)
        # Save the data to an zarr file
        zarr_file = tmpdir / f"test_data_{i}.zarr"

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
            adjmat = d["adjacency_matrix"].to_numpy()
            linkmat = d["link_matrix"].to_numpy()
            max_path_f = d["max_pathlen_future"].to_numpy()
            max_path_p = d["max_pathlen_past"].to_numpy()
            adj[j, 0 : adjmat.shape[0], 0 : adjmat.shape[1]] = adjmat
            link[j, 0 : linkmat.shape[0], 0 : linkmat.shape[1]] = linkmat
            maxpathlen_future[j, 0 : max_path_f.shape[0]] = max_path_f
            max_pathlen_past[j, 0 : max_path_p.shape[0]] = max_path_p
            dimension[j] = d["dimension"]
            atomcount[j] = d["atomcount"]

        datafiles.append(zarr_file)
    return tmpdir, datafiles


@pytest.fixture
def create_data_zarr(create_data_zarr_basic):
    tmpdir, datafiles = create_data_zarr_basic

    for file in datafiles:
        # Save the data to an zarr file
        store = zarr.storage.LocalStore(file, read_only=False)

        dims = zarr.open_array(store, path="dimension")

        num_samples = zarr.create_array(
            store, shape=(1,), chunks=(1,), name="num_samples", dtype="int32"
        )
        num_samples[0] = dims.shape[0]

    return tmpdir, datafiles


@pytest.fixture
def read_data():
    def reader(f: zarr.Group, idx: int, float_dtype, int_dtype, validate) -> Data:
        adj_raw = f["adjacency_matrix"][idx, :, :]
        adj_matrix = torch.tensor(adj_raw, dtype=float_dtype)
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        adj_matrix = adj_matrix.to_sparse()
        node_features = []

        # Path lengths
        max_path_future = torch.tensor(
            f["max_pathlen_future"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        max_path_past = torch.tensor(
            f["max_pathlen_past"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor
        node_features.extend([max_path_future, max_path_past])

        x = torch.cat(node_features, dim=1)

        dimension = f["dimension"][idx]

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


# models


@pytest.fixture
def gnn_block():
    # TODO: fix this
    return QG.GNNBlock(
        in_dim=16,
        out_dim=32,
        dropout=0.3,
        gnn_layer_type=torch_geometric.nn.conv.GCNConv,
        normalizer=torch.nn.BatchNorm1d,
        activation=torch.nn.ReLU,
        gnn_layer_args=[],
        gnn_layer_kwargs={"cached": False, "bias": True, "add_self_loops": True},
        norm_args=[
            32,
        ],
        norm_kwargs={"eps": 1e-5, "momentum": 0.2},
        projection_args=[16, 32],
        projection_kwargs={"bias": False},
    )


@pytest.fixture
def classifier_block():
    # TODO: Remove this
    return QG.LinearSequential(
        dims=[(32, 24), (24, 12), (12, 3)],
        activations=[torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        linear_kwargs=[{"bias": True}, {"bias": True}, {"bias": False}],
        activation_kwargs=[{"inplace": False}, {}, {}],
    )


@pytest.fixture
def pooling_layer():
    return torch_geometric.nn.global_mean_pool


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


@pytest.fixture
def model_config_eval():
    return {
        "encoder_type": QG.SequentialModel,
        "encoder_args": [
            "x, edge_index -> x3",
            [
                (
                    "x, edge_index -> x1",
                    "torch_geometric.nn.conv.GCNConv",
                    [
                        2,
                        8,
                    ],
                    {"improved": True, "cached": False, "add_self_loops": True},
                ),
                (
                    "x1 -> x2",
                    "torch_geometric.nn.norm.Batchnorm",
                    [
                        8,
                    ],
                    {"eps": 2e-5, "momentum": 0.2},
                ),
                (
                    "x2 -> x3",
                    "torch.nn.ReLU",
                    [],
                    {
                        "inplace": False,
                    },
                ),
                (
                    "x2, edge_index -> x3",
                    "torch_geometric.nn.conv.GCNConv",
                    [
                        8,
                        12,
                    ],
                    {"improved": True, "cached": False, "add_self_loops": True},
                ),
            ],
        ],
        "encoder_kwargs": {},
        "pooling_layers": [
            ("torch_geometric.nn.aggr.MeanAggregation", [], {}),
        ],
        "downstream_tasks": {
            0: [
                (
                    "torch_geometric.nn.sequential.Sequential",
                    [
                        "x, edge_index -> x_",
                        (
                            "x, edge_index -> x1",
                            "torch_geometric.nn.dense.Linear",
                            [8, 12],
                            {
                                "bias": True,
                            },
                            "x1 -> x2",
                            "torch.nn.ReLU",
                            [],
                            {
                                "inplace": False,
                            },
                            "x2, edge_index -> x3",
                            "torch_geometric.nn.dense.Linear",
                            [12, 20],
                            {
                                "bias": True,
                            },
                            "x3 -> x4",
                            "torch.nn.ReLU",
                            [],
                            {
                                "inplace": False,
                            },
                            "x4, edge_index -> x5",
                            "torch_geometric.nn.dense.Linear",
                            [20, 8],
                            {
                                "bias": True,
                            },
                        ),
                    ],
                    {},
                ),
            ],
        },
        "aggregate_pooling_type": "torch.nn.Identity",
        "aggregate_pooling_args": [],
        "aggregate_pooling_kwargs": {},
        "graph_features_net_type": None,
        "graph_features_net_args": None,
        "graph_features_net_kwargs": None,
        "aggregate_graph_features_type": None,
        "aggregate_graph_features_args": None,
        "aggregate_graph_features_kwargs": None,
        "active_tasks": {
            0: True,
        },
    }


@pytest.fixture
def gnn_model_eval(model_config_eval):
    """Fixture to create a GNNModel for evaluation."""

    model = QG.GNNModel.from_config(
        model_config_eval,
    )
    model.eval()
    return model
