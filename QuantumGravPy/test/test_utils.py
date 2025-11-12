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
    assert torch.nn.BatchNorm1d in QG.gnn_block.utils.normalizer_layers_names
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


def test_verify_config_node():
    """Test verification of config node."""
    valid_cfg = {
        "type": "gcn",
        "args": [16, 32],
        "kwargs": {
            "activation": "relu",
        },
    }
    assert QG.utils.verify_config_node(valid_cfg) is True

    invalid_cfg_missing_key = {
        "args": [16, 32],
        "kwargs": {
            "activation": "relu",
        },
    }
    assert QG.utils.verify_config_node(invalid_cfg_missing_key) is False

    invalid_cfg = {
        "type": "gcn",
        "kwargs": {
            "activation": "relu",
        },
    }
    assert QG.utils.verify_config_node(invalid_cfg) is False

    invalid_cfg = {
        "type": "gcn",
        "args": [16, 32],
    }
    assert QG.utils.verify_config_node(invalid_cfg) is False

    wrong_type_cfg = {
        "type": "gcn",
        "args": [16, 32],
        "kwargs": [1, 2, 3],
    }

    assert QG.utils.verify_config_node(wrong_type_cfg) is False

    wrong_type_cfg = {
        "type": "gcn",
        "args": "not a list",
        "kwargs": {
            "activation": "relu",
        },
    }
    assert QG.utils.verify_config_node(wrong_type_cfg) is False

    assert QG.utils.verify_config_node("not a dict") is False


def test_assign_at_path():
    testdict = {
        "a": {
            "b": {
                "c": 3,
            }
        },
        "d": 42,
    }

    QG.utils.assign_at_path(
        testdict,
        [
            "a",
            "b",
            "c",
        ],
        12,
    )
    assert testdict["a"]["b"]["c"] == 12

    QG.utils.assign_at_path(
        testdict,
        [
            "a",
            "b",
        ],
        {"x": 42},
    )

    assert testdict["a"]["b"] == {"x": 42}

    with pytest.raises(KeyError):
        QG.utils.assign_at_path(testdict, ["v", "z"], 12)


def test_get_at_path():
    testdict = {
        "a": {
            "b": {
                "c": 3,
            }
        },
        "d": 42,
        "r": {3: "v"},
    }

    assert QG.utils.get_at_path(testdict, ["a", "b", "c"]) == 3

    assert QG.utils.get_at_path(testdict, ["r", 3]) == "v"

    with pytest.raises(KeyError):
        QG.utils.get_at_path(
            testdict,
            ["x", "v"],
        )


def test_import_and_get():
    """Test the import_and_get function."""
    # Test importing a standard library class
    result = QG.utils.import_and_get("torch.nn.Linear")
    assert result is torch.nn.Linear

    # Test importing a function
    result = QG.utils.import_and_get("torch.cat")
    assert result is torch.cat

    # Test importing from torch_geometric
    result = QG.utils.import_and_get("torch_geometric.nn.global_mean_pool")
    assert result is global_mean_pool

    # Test invalid module path
    with pytest.raises(ValueError, match="Importing module .* unsuccessful"):
        QG.utils.import_and_get("nonexistent.module.Class")

    # Test invalid object name
    with pytest.raises(ValueError, match="Could not load name .* from"):
        QG.utils.import_and_get("torch.nn.NonExistentClass")


def test_register_evaluation_function():
    """Test registration of evaluation functions."""

    def dummy_eval_func(x, y):
        return x + y

    # Register a new evaluation function
    QG.register_evaluation_function("test_eval", dummy_eval_func)
    assert "test_eval" in QG.utils.evaluation_funcs
    assert QG.utils.evaluation_funcs["test_eval"] is dummy_eval_func

    # Test registering a class
    class DummyEvaluator:
        def __call__(self, x, y):
            return x * y

    QG.register_evaluation_function("test_eval_class", DummyEvaluator)
    assert "test_eval_class" in QG.utils.evaluation_funcs
    assert QG.utils.evaluation_funcs["test_eval_class"] is DummyEvaluator


def test_get_evaluation_function():
    """Test retrieval of registered evaluation functions."""

    def sample_func():
        return 42

    QG.register_evaluation_function("sample_eval", sample_func)

    # Test getting existing function
    retrieved = QG.get_evaluation_function("sample_eval")
    assert retrieved is sample_func
    assert retrieved() == 42

    # Test getting non-existent function
    not_registered = QG.get_evaluation_function("non_existent_eval")
    assert not_registered is None


def test_list_evaluation_functions():
    """Test listing registered evaluation functions."""

    # Clear and add some test functions
    initial_count = len(QG.list_evaluation_functions())

    QG.register_evaluation_function("test_list_1", lambda x: x)
    QG.register_evaluation_function("test_list_2", lambda x: x * 2)

    funcs = QG.list_evaluation_functions()
    assert isinstance(funcs, list)
    assert "test_list_1" in funcs
    assert "test_list_2" in funcs
    assert len(funcs) >= initial_count + 2


def test_list_registered_gnn_layers():
    """Test listing of registered GNN layers."""
    layers = QG.list_registered_gnn_layers()
    assert isinstance(layers, list)
    assert "gcn" in layers
    assert "gat" in layers
    assert "sage" in layers
    assert len(layers) > 0


def test_list_registered_normalizers():
    """Test listing of registered normalizer layers."""
    normalizers = QG.list_registered_normalizers()
    assert isinstance(normalizers, list)
    assert "batch_norm" in normalizers
    assert "layer_norm" in normalizers
    assert "identity" in normalizers
    assert len(normalizers) > 0


def test_list_registered_activations():
    """Test listing of registered activation layers."""
    activations = QG.list_registered_activations()
    assert isinstance(activations, list)
    assert "relu" in activations
    assert "sigmoid" in activations
    assert "tanh" in activations
    assert len(activations) > 0


def test_register_graph_features_aggregation():
    """Test registration of graph features aggregation functions."""

    def custom_agg(tensors):
        return torch.mean(torch.stack(tensors), dim=0)

    QG.register_graph_features_aggregation("test_agg", custom_agg)
    assert "test_agg" in QG.gnn_block.utils.graph_features_aggregations

    with pytest.raises(ValueError):
        QG.register_graph_features_aggregation("test_agg", custom_agg)


def test_register_pooling_aggregation():
    """Test registration of pooling aggregation functions."""

    def custom_pool_agg(tensors):
        return torch.sum(torch.stack(tensors), dim=0)

    QG.register_pooling_aggregation("test_pool_agg", custom_pool_agg)
    assert "test_pool_agg" in QG.gnn_block.utils.pooling_aggregations

    with pytest.raises(ValueError):
        QG.register_pooling_aggregation("test_pool_agg", custom_pool_agg)
