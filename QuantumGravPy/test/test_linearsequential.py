import pytest
import torch
import QuantumGrav as QG
import torch_geometric


@pytest.fixture
def linseq_params():
    return {
        "dims": [(10, 20), (20, 30), (30, 3)],
        "activations": [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        "linear_kwargs": [{}, {}, {}],
        "activation_kwargs": [
            {"inplace": False},
            {"inplace": False},
            {},
        ],
    }


@pytest.fixture
def linearseq(linseq_params):
    params = linseq_params
    return QG.models.LinearSequential(
        dims=params["dims"],
        activations=params["activations"],
        linear_kwargs=params["linear_kwargs"],
        activation_kwargs=params["activation_kwargs"],
    )


def test_linseq_creation(linseq_params):
    linearseq = QG.models.LinearSequential(
        dims=linseq_params["dims"],
        activations=linseq_params["activations"],
        linear_kwargs=linseq_params["linear_kwargs"],
        activation_kwargs=linseq_params["activation_kwargs"],
    )
    assert len(linearseq.layers) == 6  # 3 hidden layers + 3 activations
    assert isinstance(linearseq.layers[0], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.layers[1], torch.nn.ReLU)
    assert isinstance(linearseq.layers[2], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.layers[3], torch.nn.ReLU)
    assert isinstance(linearseq.layers[4], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.layers[5], torch.nn.Identity)
    assert linearseq.layers[0].in_channels == 10
    assert linearseq.layers[0].out_channels == 20
    assert linearseq.layers[2].in_channels == 20
    assert linearseq.layers[2].out_channels == 30
    assert linearseq.layers[4].in_channels == 30
    assert linearseq.layers[4].out_channels == 3


def test_linseq_broken_dim_args(linseq_params):
    with pytest.raises(
        ValueError, match="dims and activations must have the same length"
    ):
        QG.models.LinearSequential(
            dims=linseq_params["dims"],
            activations=[
                torch.nn.ReLU,
            ],  # too few
            linear_kwargs=linseq_params["linear_kwargs"],
            activation_kwargs=linseq_params["activation_kwargs"],
        )

    with pytest.raises(ValueError, match="dims must not be empty"):
        QG.models.LinearSequential(
            dims=[],
            activations=linseq_params["activations"],
            linear_kwargs=linseq_params["linear_kwargs"],
            activation_kwargs=linseq_params["activation_kwargs"],
        )


def test_linseq_forward_pass(linearseq):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    outputs = linearseq(input_tensor)

    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs.shape == (5, 3)  # First output layer shape


def test_linseq_forward_pass_with_kwargs(linseq_params):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    cl = QG.models.LinearSequential(
        dims=linseq_params["dims"],
        activations=linseq_params["activations"],
        linear_kwargs=[{"bias": True}, {"bias": False}, {"bias": True}],
        activation_kwargs=[{"inplace": False}, {"inplace": False}, {}],
    )

    outputs = cl(input_tensor)

    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs.shape == (5, 3)


def test_linseq_backward_pass(linearseq):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10

    linearseq.train()  # Set the model to training mode
    outputs = linearseq(input_tensor)

    loss = outputs.mean()
    loss.backward()
    for layer in linearseq.layers:
        if isinstance(layer, torch_geometric.nn.dense.Linear):
            assert layer.weight.grad is not None
            assert layer.bias.grad is not None


def test_linseq_from_config():
    "test construction of a linearsequential instance from config dict"
    config = {
        "dims": [[10, 20], [20, 30], [30, 3]],
        "activations": [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        "linear_kwargs": [{}, {}, {}],
        "activation_kwargs": [
            {"inplace": False},
            {"inplace": False},
            {},
        ],
    }

    linearseq = QG.models.LinearSequential.from_config(config)

    assert isinstance(linearseq, QG.models.LinearSequential)
    assert len(linearseq.layers) == 6
    assert linearseq.layers[0].in_channels == 10
    assert linearseq.layers[0].out_channels == 20
    assert linearseq.layers[2].in_channels == 20
    assert linearseq.layers[2].out_channels == 30
    assert linearseq.layers[4].in_channels == 30
    assert linearseq.layers[4].out_channels == 3


def test_linseq_save_load(linearseq, tmp_path):
    "test saving of the linearseq model"
    linearseq.save(tmp_path / "model.pt")

    assert (tmp_path / "model.pt").exists()

    loaded_linseq = QG.models.LinearSequential.load(tmp_path / "model.pt")

    assert loaded_linseq.state_dict().keys() == linearseq.state_dict().keys()
    for k in loaded_linseq.state_dict().keys():
        assert torch.equal(loaded_linseq.state_dict()[k], linearseq.state_dict()[k])

    # Test that the loaded model produces the same outputs as the original model
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    linearseq.eval()
    loaded_linseq.eval()

    original_outputs = linearseq(input_tensor)
    loaded_outputs = loaded_linseq(input_tensor)
    assert torch.allclose(original_outputs, loaded_outputs, atol=1e-8)
