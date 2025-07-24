import pytest
import juliacall as jcall
from pathlib import Path
import h5py

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

    for i in range(3):
        data = jl_generator(5)
        # Save the data to an HDF5 file
        hdf5_file = tmpdir / f"test_data_{i}.h5"

        with h5py.File(hdf5_file, "w") as f:
            f.create_dataset("adjacency_matrix", (len(data), 15, 15), dtype="float32")
            f.create_dataset("link_matrix", (len(data), 15, 15), dtype="float32")
            f.create_dataset("max_pathlen_future", (len(data), 15), dtype="float32")
            f.create_dataset("max_pathlen_past", (len(data), 15), dtype="float32")
            f.create_dataset("manifold", (len(data),), dtype="int32")
            f.create_dataset("boundary", (len(data),), dtype="int32")
            f.create_dataset("dimension", (len(data),), dtype="int32")
            f.create_dataset("atomcount", (len(data),), dtype="int32")

        with h5py.File(hdf5_file, "a") as f:
            for j, d in enumerate(data):
                adj = d["adjacency_matrix"].to_numpy()
                link = d["link_matrix"].to_numpy()
                max_path_f = d["max_pathlen_future"].to_numpy()
                max_path_p = d["max_pathlen_past"].to_numpy()
                f["adjacency_matrix"][j, 0 : adj.shape[0], 0 : adj.shape[1]] = adj
                f["link_matrix"][j, 0 : link.shape[0], 0 : link.shape[1]] = link
                f["max_pathlen_future"][j, 0 : max_path_f.shape[0]] = max_path_f
                f["max_pathlen_past"][j, 0 : max_path_p.shape[0]] = max_path_p
                f["manifold"][j] = d["manifold"]
                f["boundary"][j] = d["boundary"]
                f["dimension"][j] = d["dimension"]
                f["atomcount"][j] = d["atomcount"]
            f["num_causal_sets"] = len(data)

            datafiles.append(hdf5_file)
    return tmpdir, datafiles


@pytest.fixture
def read_data():
    def reader(f: h5py.File, idx: int, float_dtype, int_dtype, validate) -> Data:
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

        manifold = f["manifold"][idx]
        boundary = f["boundary"][idx]
        dimension = f["dimension"][idx]

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight.unsqueeze(1),
            y=torch.stack(
                [
                    torch.tensor(manifold),
                    torch.tensor(boundary),
                    torch.tensor(dimension),
                ],
                dim=0,
            ),
        )

        if validate and not data.validate():
            raise ValueError("Data validation failed.")
        return data

    return reader
