import pytest
import QuantumGrav as QG
from jsonschema import ValidationError

import torch_geometric
import torch


@pytest.fixture
def config():
    return {
        "layers": [
            [
                torch_geometric.nn.conv.SAGEConv,
                [2, 8],
                {"root_weight": False, "bias": False},
                "x, edge_index -> x1",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 8],
                {"root_weight": False, "bias": False},
                "x1, edge_index -> x2",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 8],
                {"root_weight": False, "bias": True},
                "x2, edge_index -> x3",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 4],
                {"root_weight": False, "bias": True},
                "x3, edge_index -> x4",
            ],
        ],
        "forward_signature": "x, edge_index, batch",
    }


@pytest.fixture
def config_missing_args():
    return {
        "layers": [
            [
                torch_geometric.nn.conv.SAGEConv,
                # no args here
                {"root_weight": False, "bias": False},
                "x, edge_index -> x1",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 8],
                {"root_weight": False, "bias": False},
                "x1, edge_index -> x2",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 8],
                {"root_weight": False, "bias": True},
                "x2, edge_index -> x3",
            ],
            [
                torch_geometric.nn.conv.SAGEConv,
                [8, 4],
                {"root_weight": False, "bias": True},
                "x3, edge_index -> x4",
            ],
        ],
        "forward_signature": "x, edge_index, batch",
    }


def test_sequential_construction():
    model = QG.models.Sequential(
        [
            (
                torch_geometric.nn.conv.SAGEConv,
                [2, 8],
                {"root_weight": False, "bias": False},
                "x, edge_index -> x1",
            ),
            (torch.nn.Dropout, [], {"p": 0.5}, "x1 -> x2"),
            (
                torch_geometric.nn.conv.SAGEConv,
                [8, 4],
                {"root_weight": False, "bias": False},
                "x, edge_index -> x1",
            ),
        ],
        "x, edge_index",
    )

    # check that the layer number is right
    assert len(model.layers) == 3
    assert isinstance(model.layers[0], torch_geometric.nn.conv.SAGEConv)
    assert isinstance(model.layers[1], torch.nn.Dropout)
    assert isinstance(model.layers[2], torch_geometric.nn.conv.SAGEConv)


def test_sequential_forward():
    model = QG.models.Sequential(
        [
            (
                torch_geometric.nn.conv.SAGEConv,
                [2, 8],
                {"root_weight": False, "bias": False},
                "x, edge_index -> x1",
            ),
            (torch.nn.Dropout, [], {"p": 0.5}, "x1 -> x2"),
            (
                torch_geometric.nn.conv.SAGEConv,
                [8, 4],
                {"root_weight": False, "bias": False},
                "x2, edge_index -> x3",
            ),
        ],
        "x, edge_index",
    )

    x = torch.rand((8, 2))
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)

    out = model(x, edge_index)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (8, 4)


def test_sequential_from_config(config):
    model = QG.models.Sequential.from_config(config)
    # check that the layer number is right
    assert len(model.layers) == 4
    assert isinstance(model.layers[0], torch_geometric.nn.conv.SAGEConv)
    assert isinstance(model.layers[1], torch_geometric.nn.conv.SAGEConv)
    assert isinstance(model.layers[2], torch_geometric.nn.conv.SAGEConv)
    assert isinstance(model.layers[2], torch_geometric.nn.conv.SAGEConv)


def test_sequential_from_config_broken(config_missing_args):
    with pytest.raises(ValidationError):
        QG.models.Sequential.from_config(config_missing_args)
