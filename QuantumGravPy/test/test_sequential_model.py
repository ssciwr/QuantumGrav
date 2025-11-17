import QuantumGrav as QG
import torch
import pytest
import numpy as np


@pytest.fixture
def gnn_block():
    """Fixture to provide configuration for GNNBlock."""
    input_sig = "x, edge_index -> x4"
    layers = [
        (
            # gcn layer
            "x, edge_index -> x1",
            "tgnn.conv.GCNConv",
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
            "torch.nn.ReLU",
            [],
            {"inplace": False},
        ),
        (
            # batch norm
            "x2 -> x3",
            "tgnn.norm.BatchNorm",
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
            "torch.nn.Dropout",
            [],
            {"p": 0.3, "inplace": False},
        ),
    ]

    return QG.SequentialModel(input_sig, layers)


@pytest.fixture
def gnnblock_conf():
    return {
        "input_sig": "x, edge_index -> x4",
        "layer_0": {
            "signature": "x, edge_index -> x1",
            "type": "tgnn.conv.GCNConv",
            "args": [16, 32],
            "kwargs": {
                "improved": True,
                "cached": False,
                "add_self_loops": False,
                "normalize": True,
                "bias": True,
            },
        },
        "layer_1": {
            "signature": "x1 -> x2",
            "type": "torch.nn.ReLU",
            "args": [],
            "kwargs": {"inplace": False},
        },
        "layer_2": {
            "signature": "x2 -> x3",
            "type": "tgnn.norm.BatchNorm",
            "args": [
                32,
            ],
            "kwargs": {
                "eps": 1e-05,
                "momentum": 0.1,
                "affine": True,
                "track_running_stats": True,
                "allow_single_element": False,
            },
        },
        "layer_3": {
            "signature": "x3 -> x4",
            "type": "torch.nn.Dropout",
            "args": [],
            "kwargs": {"p": 0.3, "inplace": False},
        },
    }


def test_gnn_block_initialization(gnn_block):
    assert gnn_block.input_sig == "x, edge_index -> x4"
    assert gnn_block.layerspecs == 3  # TODO
    assert gnn_block.layers == 3  # TODO
    assert 3 == 6


def test_gnn_block_properties(gnn_block):
    assert 3 == 6


def test_gnn_block_forward(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    y = gnn_block.forward(x, edge_index)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros
    assert 3 == 6


def test_gnn_block_forward_with_edge_weight(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

    y_simple = gnn_block.forward(x, edge_index)
    y = gnn_block.forward(x, edge_index, edge_weight=edge_weight)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert not torch.equal(y, y_simple)  # Ensure edge weights affect output
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros
    assert 3 == 6


def test_gnn_block_backward(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    gnn_block.train()
    y = gnn_block.forward(x, edge_index)
    loss = y.sum()  # simple loss for testing
    loss.backward()

    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert not torch.equal(y, x)  # Ensure output is not equal to input
    assert torch.count_nonzero(y).item() > 0  # Ensure output is not all zeros
    assert gnn_block.projection.weight.grad is not None
    assert gnn_block.conv.lin.weight.grad is not None
    assert gnn_block.normalizer.weight.grad is not None
    assert 3 == 6


def test_gnn_block_from_config(gnn_block_config):
    "test construction of model from config"
    gnn_block = QG.SequentialModel.from_config(gnn_block_config)
    assert gnn_block.layerspecs == 3
    assert 3 == 6


def test_gnn_block_to_config(gnn_block):
    config = gnn_block.to_config()

    assert config["in_dim"] == gnn_block.in_dim
    assert config["out_dim"] == gnn_block.out_dim
    assert config["dropout"] == gnn_block.dropout_p
    assert config["with_skip"] == gnn_block.with_skip
    assert config["gnn_layer_type"] == "gcn"
    assert config["normalizer"] == "batch_norm"
    assert config["activation"] == "relu"
    assert config["gnn_layer_args"] == gnn_block.gnn_layer_args
    assert config["gnn_layer_kwargs"] == gnn_block.gnn_layer_kwargs
    assert config["norm_args"] == gnn_block.norm_args
    assert config["norm_kwargs"] == gnn_block.norm_kwargs
    assert config["activation_args"] == gnn_block.activation_args
    assert config["activation_kwargs"] == gnn_block.activation_kwargs
    assert config["projection_args"] == gnn_block.projection_args
    assert config["projection_kwargs"] == gnn_block.projection_kwargs
    assert 3 == 6


def test_gnn_block_config_roundtrip(gnn_block):
    assert 3 == 6


def test_gnn_block_save_load(gnn_block, tmp_path):
    "test saving and loading of the gnn_block"

    gnn_block.save(tmp_path / "model.pt")
    assert (tmp_path / "model.pt").exists()

    loaded_gnn_block = QG.SequentialModel.load(tmp_path / "model.pt")
    assert loaded_gnn_block.state_dict().keys() == gnn_block.state_dict().keys()
    for k in loaded_gnn_block.state_dict().keys():
        assert torch.equal(loaded_gnn_block.state_dict()[k], gnn_block.state_dict()[k])

    gnn_block.eval()
    loaded_gnn_block.eval()
    x = torch.tensor(np.random.uniform(0, 1, (5, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y = gnn_block.forward(x, edge_index)
    y_loaded = loaded_gnn_block.forward(x, edge_index)
    assert y.shape == y_loaded.shape
    assert torch.allclose(y, y_loaded, atol=1e-8)
    assert 3 == 6
