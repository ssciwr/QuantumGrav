import QuantumGrav as QG
import pytest
import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_mean_pool


def test_register_pooling_layer():
    """Test the registration of pooling layers."""
    layer = torch.nn.MaxPool1d(2)
    QG.register_pooling_layer("test_pooling", layer)
    assert "test_pooling" in QG.gnn_block.utils.pooling_layers

    with pytest.raises(ValueError):
        QG.register_pooling_layer("test_pooling", layer)


def test_get_registered_pooling_layer():
    """Test retrieval of registered pooling layers."""
    layer = QG.get_registered_pooling_layer("mean")
    assert layer is not None
    assert layer is global_mean_pool

    not_registered_layer = QG.get_registered_pooling_layer("non_existent")
    assert not_registered_layer is None


def test_list_registered_pooling_layers():
    """Test listing of registered pooling layers."""
    layers = QG.list_registered_pooling_layers()
    assert isinstance(layers, list)
    assert "mean" in layers
    assert "max" in layers
    assert "sum" in layers
    assert len(layers) > 0  # Ensure there are some registered layers


def test_register_gnn_layer():
    """Test the registration of GNN layers."""
    layer = torch.nn.Linear
    QG.register_gnn_layer("test_layer", layer)
    assert "test_layer" in QG.gnn_block.utils.gnn_layers

    with pytest.raises(ValueError):
        QG.register_gnn_layer("test_layer", layer)


def test_register_activation():
    """Test the registration of activation layers."""
    activation = torch.nn.ReLU
    QG.register_activation("test_activation", activation)
    assert "test_activation" in QG.gnn_block.utils.activation_layers

    with pytest.raises(ValueError):
        QG.register_activation("test_activation", activation)


def test_register_normalizer():
    """Test the registration of normalizer layers."""
    normalizer = torch.nn.BatchNorm1d
    QG.register_normalizer("test_normalizer", normalizer)
    assert "test_normalizer" in QG.gnn_block.utils.normalizer_layers

    with pytest.raises(ValueError):
        QG.register_normalizer("test_normalizer", normalizer)


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


def test_get_registered_pooling_aggregation():
    """Test retrieval of registered pooling aggregation functions."""
    cat = QG.get_pooling_aggregation("cat0")
    assert cat == torch.cat

    not_registered_mean = QG.get_pooling_aggregation("non_existent")
    assert not_registered_mean is None


def test_get_registered_graph_features_aggregation():
    """Test retrieval of registered graph features aggregation functions."""

    cat0 = QG.get_graph_features_aggregation("cat0")
    assert cat0 == torch.cat

    not_registered_mean = QG.get_graph_features_aggregation("non_existent")
    assert not_registered_mean is None
