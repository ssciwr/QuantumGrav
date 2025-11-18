import QuantumGrav as QG
import torch
import torch_geometric.nn as tgnn
import pytest
import numpy as np


@pytest.fixture
def seqmodel():
    """Fixture to provide configuration for seqmodel."""
    input_signature = "x, edge_index"
    layers = [
        (
            # gcn layer
            "x, edge_index -> x1",
            tgnn.conv.GCNConv,
            [16, 32],
            {
                "improved": True,
                "cached": False,
                "add_self_loops": False,
                "normalize": True,
                "bias": True,
            },
        ),
        (
            # activation
            "x1 -> x2",
            torch.nn.ReLU,
            [],
            {"inplace": False},
        ),
        (
            # batch norm
            "x2 -> x3",
            tgnn.norm.BatchNorm,
            [
                32,
            ],
            {
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
                "allow_single_element": False,
            },
        ),
        (
            # dropout
            "x3 -> x4",
            torch.nn.Dropout,
            [],
            {"p": 0.3, "inplace": False},
        ),
    ]

    return QG.SequentialModel(input_signature, layers)


@pytest.fixture
def seqmodel_config():
    return {
        "input_signature": "x, edge_index",
        "layer_specs": [
            [
                "x, edge_index -> x1",
                tgnn.conv.GCNConv,
                [16, 32],
                {
                    "improved": True,
                    "cached": False,
                    "add_self_loops": False,
                    "normalize": True,
                    "bias": True,
                },
            ],
            [
                "x1 -> x2",
                torch.nn.ReLU,
                [],
                {"inplace": False},
            ],
            [
                "x2 -> x3",
                tgnn.norm.BatchNorm,
                [
                    32,
                ],
                {
                    "eps": 1e-05,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": True,
                    "allow_single_element": False,
                },
            ],
            [
                "x3 -> x4",
                torch.nn.Dropout,
                [],
                {"p": 0.3, "inplace": False},
            ],
        ],
    }


def test_seqmodel_initialization(seqmodel):
    assert seqmodel.input_signature == "x, edge_index"
    assert seqmodel.layerspecs == [
        [
            # gcn layer
            "x, edge_index -> x1",
            tgnn.conv.GCNConv,
            [16, 32],
            {
                "improved": True,
                "cached": False,
                "add_self_loops": False,
                "normalize": True,
                "bias": True,
            },
        ],
        [
            # activation
            "x1 -> x2",
            torch.nn.ReLU,
            [],
            {"inplace": False},
        ],
        [
            # batch norm
            "x2 -> x3",
            tgnn.norm.BatchNorm,
            [
                32,
            ],
            {
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
                "allow_single_element": False,
            },
        ],
        [
            # dropout
            "x3 -> x4",
            torch.nn.Dropout,
            [],
            {"p": 0.3, "inplace": False},
        ],
    ]
    assert len(seqmodel.layers) == len(seqmodel.layerspecs)
    assert isinstance(
        seqmodel.layers[0],
        tgnn.conv.GCNConv,
    )
    assert isinstance(
        seqmodel.layers[1],
        torch.nn.ReLU,
    )
    assert isinstance(seqmodel.layers[2], tgnn.norm.BatchNorm)
    assert isinstance(
        seqmodel.layers[3],
        torch.nn.Dropout,
    )


def test_seqmodel_forward(seqmodel):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    y = seqmodel.forward(x, edge_index)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros


def test_seqmodel_forward_with_edge_weight(seqmodel):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

    y_simple = seqmodel.forward(x, edge_index)
    y = seqmodel.forward(x, edge_index, edge_weight=edge_weight)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert not torch.equal(y, y_simple)  # Ensure edge weights affect output
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros


def test_seqmodel_backward(seqmodel):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    seqmodel.train()
    y = seqmodel.forward(x, edge_index)
    loss = y.sum()  # simple loss for testing
    loss.backward()

    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros

    for name, param in seqmodel.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.all(param.grad == 0), f"Parameter {name} has zero gradient"


def test_seqmodel_from_config(seqmodel_config):
    "test construction of model from config"
    seqmodel = QG.SequentialModel.from_config(seqmodel_config)
    assert seqmodel.layerspecs == [
        [
            # gcn layer
            "x, edge_index -> x1",
            tgnn.conv.GCNConv,
            [16, 32],
            {
                "improved": True,
                "cached": False,
                "add_self_loops": False,
                "normalize": True,
                "bias": True,
            },
        ],
        [
            # activation
            "x1 -> x2",
            torch.nn.ReLU,
            [],
            {"inplace": False},
        ],
        [
            # batch norm
            "x2 -> x3",
            tgnn.norm.BatchNorm,
            [
                32,
            ],
            {
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
                "allow_single_element": False,
            },
        ],
        [
            # dropout
            "x3 -> x4",
            torch.nn.Dropout,
            [],
            {"p": 0.3, "inplace": False},
        ],
    ]
    assert len(seqmodel.layers) == len(seqmodel.layerspecs)
    assert isinstance(
        seqmodel.layers[0],
        tgnn.conv.GCNConv,
    )
    assert isinstance(
        seqmodel.layers[1],
        torch.nn.ReLU,
    )
    assert isinstance(seqmodel.layers[2], tgnn.norm.BatchNorm)
    assert isinstance(
        seqmodel.layers[3],
        torch.nn.Dropout,
    )


def test_seqmodel_to_config(seqmodel):
    config = seqmodel.to_config()
    assert config["input_signature"] == "x, edge_index"
    assert config["layer_specs"] == [
        [
            # gcn layer
            "x, edge_index -> x1",
            tgnn.conv.GCNConv,
            [16, 32],
            {
                "improved": True,
                "cached": False,
                "add_self_loops": False,
                "normalize": True,
                "bias": True,
            },
        ],
        [
            # activation
            "x1 -> x2",
            torch.nn.ReLU,
            [],
            {"inplace": False},
        ],
        [
            # batch norm
            "x2 -> x3",
            tgnn.norm.BatchNorm,
            [
                32,
            ],
            {
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
                "allow_single_element": False,
            },
        ],
        [
            # dropout
            "x3 -> x4",
            torch.nn.Dropout,
            [],
            {"p": 0.3, "inplace": False},
        ],
    ]


def test_seqmodel_config_roundtrip(seqmodel):
    config = seqmodel.to_config()
    reconstructed_model = QG.SequentialModel.from_config(config)
    assert seqmodel.input_signature == reconstructed_model.input_signature
    assert seqmodel.layerspecs == reconstructed_model.layerspecs


def test_seqmodel_save_load(seqmodel, tmp_path):
    "test saving and loading of the seqmodel"

    seqmodel.save(tmp_path / "model.pt")
    assert (tmp_path / "model.pt").exists()

    loaded_seqmodel = QG.SequentialModel.load(tmp_path / "model.pt")
    assert loaded_seqmodel.state_dict().keys() == seqmodel.state_dict().keys()
    for k in loaded_seqmodel.state_dict().keys():
        assert torch.equal(loaded_seqmodel.state_dict()[k], seqmodel.state_dict()[k])

    seqmodel.eval()
    loaded_seqmodel.eval()
    x = torch.tensor(np.random.uniform(0, 1, (5, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y = seqmodel.forward(x, edge_index)
    y_loaded = loaded_seqmodel.forward(x, edge_index)
    assert y.shape == y_loaded.shape
    assert torch.allclose(y, y_loaded, atol=1e-8)
