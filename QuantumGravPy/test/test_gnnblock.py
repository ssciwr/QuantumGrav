import QuantumGrav as QG
import torch
import pytest
import torch_geometric.nn as tgnn
import numpy as np


@pytest.fixture
def gnn_block_config():
    """Fixture to provide configuration for GNNBlock."""
    return {
        "in_channels": 16,
        "out_channels": 32,
        "dropout": 0.3,
        "gcn_type": "gcn",
        "normalizer": "batch_norm",
        "activation": "relu",
        "norm_args": [
            32,
        ],
        "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
        "gcn_kwargs": {"cached": False, "bias": True, "add_self_loops": True},
    }


@pytest.fixture
def gnn_block():
    return QG.GNNBlock(
        in_channels=16,
        out_channels=32,
        dropout=0.3,
        gcn_type=tgnn.conv.GCNConv,
        normalizer=torch.nn.BatchNorm1d,
        activation=torch.nn.ReLU,
        gcn_args=[],
        gcn_kwargs={"cached": False, "bias": True, "add_self_loops": True},
        norm_args=[
            32,
        ],
        norm_kwargs={"eps": 1e-5, "momentum": 0.2},
    )


def test_gnn_block_initialization(gnn_block):
    assert gnn_block.dropout_p == 0.3
    assert gnn_block.in_channels == 16
    assert gnn_block.out_channels == 32
    assert isinstance(gnn_block.dropout, torch.nn.Dropout)
    assert isinstance(gnn_block.conv, tgnn.conv.GCNConv)
    assert isinstance(gnn_block.normalizer, torch.nn.BatchNorm1d)
    assert isinstance(gnn_block.activation, torch.nn.ReLU)
    assert isinstance(gnn_block.projection, torch.nn.Linear)

    test_gnnblock_flat = QG.GNNBlock(
        in_channels=16,
        out_channels=16,
        dropout=0.3,
        gcn_type=tgnn.conv.GCNConv,
        normalizer=torch.nn.BatchNorm1d,
        activation=torch.nn.ReLU,
        gcn_args=[],
        gcn_kwargs={"cached": False, "bias": True, "add_self_loops": True},
        norm_args=[
            16,
        ],
        norm_kwargs={"eps": 1e-5, "momentum": 0.2},
    )
    assert isinstance(test_gnnblock_flat.projection, torch.nn.Identity)


def test_gnn_block_properties(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # run conv
    y = gnn_block.conv(x, edge_index)
    assert y.shape == (10, 32)
    assert torch.count_nonzero(x).item() > 0  # ensure input is not all zeros

    y_normalized = gnn_block.normalizer(y)
    assert y_normalized.shape == (10, 32)
    assert not torch.isnan(y_normalized).any()
    assert not torch.isinf(y_normalized).any()

    # run projection
    y_proj = gnn_block.projection(x)
    assert y_proj.shape == (10, 32)
    assert not torch.isnan(y_proj).any()
    assert not torch.isinf(y_proj).any()

    # run dropout
    zeroed_og = (y == 0).sum().item()
    assert zeroed_og == 0  # ensure no zeros before dropout
    y_dropout = gnn_block.dropout(y)
    assert y_dropout.shape == (10, 32)

    # Count number of zeroed elements (from dropout)
    zeroed = (y_dropout == 0).sum().item()
    # Verify some dropout occurred but not too much
    assert 0 < zeroed < y_dropout.numel()  # Ensure some elements are zeroed out


def test_gnn_block_forward(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    y = gnn_block.forward(x, edge_index)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_gnn_block_forward_with_edge_weight(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5], dtype=torch.float)

    y = gnn_block.forward(x, edge_index, edge_weight=edge_weight)
    assert y.shape == (10, 32)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_gnn_block_backward(gnn_block):
    x = torch.tensor(np.random.uniform(0, 1, (10, 16)), dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    gnn_block.train()
    y = gnn_block.forward(x, edge_index)
    loss = y.sum()  # simple loss for testing
    loss.backward()

    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_gnn_block_from_config(gnn_block_config):
    gnn_block = QG.GNNBlock.from_config(gnn_block_config)
    assert gnn_block.in_channels == gnn_block_config["in_channels"]
    assert gnn_block.out_channels == gnn_block_config["out_channels"]
    assert gnn_block.dropout_p == gnn_block_config["dropout"]
    assert isinstance(gnn_block.conv, tgnn.conv.GCNConv)
    assert isinstance(gnn_block.normalizer, torch.nn.BatchNorm1d)
    assert isinstance(gnn_block.activation, torch.nn.ReLU)
    assert isinstance(gnn_block.projection, torch.nn.Linear)
