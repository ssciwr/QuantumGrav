import pytest
import zarr
import numpy as np
import shutil
import QuantumGrav as QG
import torch
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

mp.set_start_method("spawn")


# data fixtures
@pytest.fixture(scope="session")
def create_data_zarr(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("test_data_quantumgrav", numbered=True)

    datafiles = []

    for i in range(3):
        data = []
        for _ in range(5):
            num_nodes = 15
            adjacency_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            link_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            max_pathlen_future = np.random.rand(num_nodes).astype("float32")
            max_pathlen_past = np.random.rand(num_nodes).astype("float32")
            dimension = np.array(
                [
                    np.random.randint(2, 10),
                ]
            )
            atomcount = np.array(
                [
                    num_nodes,
                ]
            )

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
        root = zarr.open_group(store, path="", mode="a")
        for j, d in enumerate(data):
            grp = root.create_group(f"cset_{j + 1}")
            for k, values in d.items():
                grp.create_array(k, data=values)

        datafiles.append(zarr_file)

        store.close()

    yield tmpdir, datafiles

    # remove created files again
    for file in datafiles:
        if file.exists():
            shutil.rmtree(file)
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture(scope="session")
def create_data_zarr_zip(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("test_data_quantumgrav_zip", numbered=True)

    datafiles = []

    for i in range(3):
        data = []
        for _ in range(5):
            num_nodes = 15
            adjacency_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            link_matrix = np.random.rand(num_nodes, num_nodes).astype("float32")
            max_pathlen_future = np.random.rand(num_nodes).astype("float32")
            max_pathlen_past = np.random.rand(num_nodes).astype("float32")
            dimension = np.array([np.random.randint(2, 10)])
            atomcount = np.array([num_nodes])

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

        zarr_file = tmpdir / f"test_data_{i}.zip"
        store = zarr.storage.ZipStore(zarr_file, mode="w")
        root = zarr.open_group(store, path="", mode="w")
        for j, d in enumerate(data):
            grp = root.create_group(f"cset_{j + 1}")
            for k, values in d.items():
                grp.create_array(k, data=values)
        store.close()
        datafiles.append(zarr_file)

    yield tmpdir, datafiles

    for file in datafiles:
        if file.exists():
            file.unlink()
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture
def pre_transform():
    global pre_transform_function

    def _pre_transform_func(datadict: dict) -> Data:
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

    pre_transform_function = _pre_transform_func
    return pre_transform_function


@pytest.fixture
def make_dataset(create_data_zarr, pre_transform):
    datadir, datafiles = create_data_zarr
    pre_transform_function = pre_transform
    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
        transform=lambda x: x,
        pre_transform=pre_transform_function,
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
        model:
            name: test_model
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
                    target: model.foo[1].x
                    values: [-10, -20]

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
