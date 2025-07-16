import pytest
import juliacall as jcall
import h5py
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


@pytest.fixture(scope="module")
def basic_transform():
    def transform(raw: dict[Any, Any]) -> Data:
        adj_raw = raw["adjacency_matrix"]
        adj_matrix = torch.tensor(adj_raw, dtype=torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        adj_matrix = adj_matrix.to_sparse()

        return Data(
            x=torch.tensor(raw["node_features"], dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor(
                [raw["manifold"], raw["boundary"], raw["dimension"]], dtype=torch.long
            ),
        )

    return transform


@pytest.fixture(scope="module")
def create_data(tmp_path):
    jl_module = jcall.newmodule("test_qg")
    jl_module.seval(
        f'using Pkg; Pkg.develop(path="../../QuantumGrav.jl")'
    )  # only for now -> get from package index later

    jl_module.seval('include("./julia_testmodule.jl")')
    data = []
    for _ in range(10):
        datadict = jl_module.seval(f"create_datapoint(42)")
        data.append(datadict)

    # Save the data to an HDF5 file
    hdf5_file = tmp_path / "test_data.h5"
    with h5py.File(hdf5_file, "w") as f:
        for k in data[0].keys():
            f.create_dataset(k, data=[d[k] for d in data], dtype="float32")


@pytest.fixture(scope="module")
def ontheflyconfig():
    onthefly_config = {
        "seed": 42,
        "n_processes": 1,
    }
    return onthefly_config
