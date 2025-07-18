import pytest
import juliacall as jcall
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


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
        classes = [raw["manifold"], raw["boundary"], raw["dimension"]]
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
            "manifold": int(raw["manifold"]),
            "boundary": int(raw["boundary"]),
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
    # Get the path to the test directory (where conftest.py is)
    test_dir = Path(__file__).parent
    # Go up to the project root (QuantumGravPy directory)
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
