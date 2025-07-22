import QuantumGrav as QG
import torch
from torch_geometric.nn.conv import GCNConv


def test_register_gnn_layer():
    """Test the registration of GNN layers."""
    layer = torch.nn.Linear
    QG.register_gnn_layer("test_layer", layer)
    assert "test_layer" in QG.gnnblock.gnn_layers


def test_register_activation():
    """Test the registration of activation layers."""
    activation = torch.nn.ReLU
    QG.register_activation("test_activation", activation)
    assert "test_activation" in QG.gnnblock.activation_layers


def test_register_normalizer():
    """Test the registration of normalizer layers."""
    normalizer = torch.nn.BatchNorm1d
    QG.register_normalizer("test_normalizer", normalizer)
    assert "test_normalizer" in QG.gnnblock.normalizer_layers


def test_get_registered_gnn_layer():
    """Test retrieval of registered GNN layers."""
    layer = QG.get_registered_gnn_layer("gcn")
    assert layer is not None
    assert layer == GCNConv

    not_registered_layer = QG.get_registered_gnn_layer("non_existent")
    assert not_registered_layer is None


def test_get_registered_normalizer():
    """Test retrieval of registered normalizer layers."""
    normalizer = QG.get_registered_normalizer("batch_norm")
    assert normalizer is not None
    assert normalizer == torch.nn.BatchNorm1d

    not_registered_normalizer = QG.get_registered_normalizer("non_existent")
    assert not_registered_normalizer is None


def test_get_registered_activation():
    """Test retrieval of registered activation layers."""
    activation = QG.get_registered_activation("relu")
    assert activation is not None
    assert activation == torch.nn.ReLU

    not_registered_activation = QG.get_registered_activation("non_existent")
    assert not_registered_activation is None
