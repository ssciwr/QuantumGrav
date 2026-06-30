import QuantumGrav as QG
import pytest
import torch


@pytest.fixture
def observable_cnn_config():
    """Fixture to provide configuration for ObservableCNNEncoder."""
    return {
        "vector_inputs": {
            "histogram_a": {
                "channels": [1, 4],
                "kernel_sizes": [3],
                "paddings": [1],
            },
            "histogram_b": {
                "channels": [2, 5],
                "kernel_sizes": [5],
                "pooling": "avgmax",
                "activation": torch.nn.GELU,
                "dropout": 0.1,
            },
        },
        "activation": torch.nn.ReLU,
        "dropout": 0.0,
        "pooling": "avg",
        "scalar_in_features": 2,
        "scalar_hidden_dims": [3],
        "scalar_activation": torch.nn.Tanh,
    }


@pytest.fixture
def observable_cnn(observable_cnn_config):
    return QG.models.ObservableCNNEncoder.from_config(observable_cnn_config)


@pytest.fixture
def observable_inputs():
    return {
        "histogram_a": torch.randn(2, 7),
        "histogram_b": torch.randn(2, 2, 11),
    }


def test_observable_cnn_initialization(observable_cnn):
    assert isinstance(observable_cnn.vector_encoders, torch.nn.ModuleDict)
    assert set(observable_cnn.vector_encoders.keys()) == {"histogram_a", "histogram_b"}
    assert isinstance(observable_cnn.vector_encoders["histogram_a"][0], torch.nn.Conv1d)
    assert isinstance(observable_cnn.vector_encoders["histogram_a"][1], torch.nn.ReLU)
    assert isinstance(observable_cnn.vector_encoders["histogram_b"][1], torch.nn.GELU)
    assert isinstance(
        observable_cnn.vector_encoders["histogram_b"][2],
        torch.nn.Dropout,
    )
    assert observable_cnn.vector_out_features == {
        "histogram_a": 4,
        "histogram_b": 10,
    }
    assert observable_cnn.scalar_out_features == 3
    assert observable_cnn.out_features == 17


def test_observable_cnn_forward(observable_cnn, observable_inputs):
    scalar_features = torch.randn(2, 2)

    y = observable_cnn.forward(observable_inputs, scalar_features)

    assert y.shape == (2, 17)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert torch.count_nonzero(y).item() > 0


def test_observable_cnn_forward_without_scalar_features():
    observable_cnn = QG.models.ObservableCNNEncoder(
        vector_inputs={
            "histogram": {
                "channels": [1, 4],
            },
        },
    )

    y = observable_cnn.forward({"histogram": torch.randn(5)})

    assert y.shape == (1, 4)
    assert isinstance(y, torch.Tensor)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_observable_cnn_backward(observable_cnn, observable_inputs):
    scalar_features = torch.randn(2, 2)

    observable_cnn.train()
    y = observable_cnn.forward(observable_inputs, scalar_features)
    loss = y.sum()
    loss.backward()

    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert observable_cnn.vector_encoders["histogram_a"][0].weight.grad is not None
    assert observable_cnn.vector_encoders["histogram_b"][0].weight.grad is not None
    assert observable_cnn.scalar_encoder[0].weight.grad is not None


def test_observable_cnn_from_config(observable_cnn_config):
    "test construction of model from config"
    observable_cnn = QG.models.ObservableCNNEncoder.from_config(observable_cnn_config)

    assert isinstance(observable_cnn, QG.models.ObservableCNNEncoder)
    assert set(observable_cnn.vector_encoders.keys()) == {"histogram_a", "histogram_b"}
    assert observable_cnn.vector_inputs["histogram_a"]["channels"] == [1, 4]
    assert observable_cnn.vector_inputs["histogram_a"]["kernel_sizes"] == [3]
    assert observable_cnn.vector_inputs["histogram_a"]["paddings"] == [1]
    assert observable_cnn.vector_inputs["histogram_b"]["channels"] == [2, 5]
    assert observable_cnn.vector_inputs["histogram_b"]["pooling"] == "avgmax"
    assert observable_cnn.vector_inputs["histogram_b"]["activation"] is torch.nn.GELU
    assert observable_cnn.scalar_activation is torch.nn.Tanh
    assert observable_cnn.out_features == 17


def test_observable_cnn_to_config(observable_cnn):
    config = observable_cnn.to_config()

    assert set(config["vector_inputs"].keys()) == {"histogram_a", "histogram_b"}
    assert config["vector_inputs"]["histogram_a"]["channels"] == [1, 4]
    assert config["vector_inputs"]["histogram_a"]["kernel_sizes"] == [3]
    assert config["vector_inputs"]["histogram_a"]["paddings"] == [1]
    assert (
        config["vector_inputs"]["histogram_a"]["activation"]
        == "torch.nn.modules.activation.ReLU"
    )
    assert config["vector_inputs"]["histogram_b"]["channels"] == [2, 5]
    assert config["vector_inputs"]["histogram_b"]["pooling"] == "avgmax"
    assert (
        config["vector_inputs"]["histogram_b"]["activation"]
        == "torch.nn.modules.activation.GELU"
    )
    assert config["scalar_in_features"] == 2
    assert config["scalar_hidden_dims"] == [3]
    assert config["scalar_activation"] == "torch.nn.modules.activation.Tanh"


def test_observable_cnn_save_load(observable_cnn, observable_inputs, tmp_path):
    "test saving and loading of the observable cnn encoder"
    observable_cnn.save(tmp_path / "model.pt")
    assert (tmp_path / "model.pt").exists()

    loaded_observable_cnn = QG.models.ObservableCNNEncoder.load(tmp_path / "model.pt")
    assert (
        loaded_observable_cnn.state_dict().keys()
        == observable_cnn.state_dict().keys()
    )
    for k in loaded_observable_cnn.state_dict().keys():
        assert torch.equal(
            loaded_observable_cnn.state_dict()[k],
            observable_cnn.state_dict()[k],
        )

    observable_cnn.eval()
    loaded_observable_cnn.eval()
    scalar_features = torch.randn(2, 2)
    y = observable_cnn.forward(observable_inputs, scalar_features)
    y_loaded = loaded_observable_cnn.forward(observable_inputs, scalar_features)
    assert y.shape == y_loaded.shape
    assert torch.allclose(y, y_loaded, atol=1e-8)


def test_observable_cnn_broken_vector_input_configs():
    with pytest.raises(ValueError, match="vector_inputs must contain at least one"):
        QG.models.ObservableCNNEncoder(vector_inputs={})

    with pytest.raises(ValueError, match="kernel_sizes must have length"):
        QG.models.ObservableCNNEncoder(
            vector_inputs={
                "histogram": {
                    "channels": [1, 4, 8],
                    "kernel_sizes": [3],
                },
            },
        )

    with pytest.raises(ValueError, match="vector input names cannot contain"):
        QG.models.ObservableCNNEncoder(
            vector_inputs={
                "nested.histogram": {
                    "channels": [1, 4],
                },
            },
        )


def test_observable_cnn_forward_error_paths(observable_cnn):
    with pytest.raises(ValueError, match="Missing vector inputs"):
        observable_cnn.forward({"histogram_a": torch.randn(2, 7)}, torch.randn(2, 2))

    with pytest.raises(ValueError, match="scalar_features must be supplied"):
        observable_cnn.forward(
            {
                "histogram_a": torch.randn(2, 7),
                "histogram_b": torch.randn(2, 2, 11),
            },
        )

    with pytest.raises(ValueError, match="Vector inputs must have shape"):
        observable_cnn.forward(
            {
                "histogram_a": torch.randn(2, 1, 7, 1),
                "histogram_b": torch.randn(2, 2, 11),
            },
            torch.randn(2, 2),
        )
