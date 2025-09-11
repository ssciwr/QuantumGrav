import pytest
import torch
import QuantumGrav as QG
import torch_geometric


@pytest.fixture
def linseq_params():
    return {
        "input_dim": 10,
        "hidden_dims": [20, 30],
        "output_dims": [2, 3],
        "activation": torch.nn.ReLU,
        "backbone_kwargs": [{}, {}],
        "output_kwargs": [{}, {}],
        "activation_kwargs": [
            {"inplace": False},
        ],
    }


@pytest.fixture
def linearseq(linseq_params):
    params = linseq_params
    return QG.linear_sequential.LinearSequential(
        input_dim=params["input_dim"],
        hidden_dims=params["hidden_dims"],
        output_dims=params["output_dims"],
        activation=params["activation"],
        backbone_kwargs=params["backbone_kwargs"],
        output_kwargs=params["output_kwargs"],
        activation_kwargs=params["activation_kwargs"],
    )


def test_linseq_creation(linseq_params):
    linearseq = QG.linear_sequential.LinearSequential(
        input_dim=linseq_params["input_dim"],
        hidden_dims=linseq_params["hidden_dims"],
        output_dims=linseq_params["output_dims"],
        activation=linseq_params["activation"],
        backbone_kwargs=linseq_params["backbone_kwargs"],
        output_kwargs=linseq_params["output_kwargs"],
        activation_kwargs=linseq_params["activation_kwargs"],
    )
    assert len(linearseq.backbone) == 4  # 2 hidden layers + 2 activations
    assert len(linearseq.output_layers) == 2
    assert isinstance(linearseq.backbone[0], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.backbone[1], torch.nn.ReLU)
    assert isinstance(linearseq.backbone[2], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.backbone[3], torch.nn.ReLU)
    assert isinstance(linearseq.output_layers[0], torch_geometric.nn.dense.Linear)
    assert isinstance(linearseq.output_layers[1], torch_geometric.nn.dense.Linear)
    assert linearseq.backbone[0].in_channels == 10
    assert linearseq.backbone[0].out_channels == 20
    assert linearseq.backbone[2].in_channels == 20
    assert linearseq.backbone[2].out_channels == 30
    assert linearseq.output_layers[0].in_channels == 30
    assert linearseq.output_layers[0].out_channels == 2
    assert linearseq.output_layers[1].in_channels == 30
    assert linearseq.output_layers[1].out_channels == 3


def test_linseq_no_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims must not be None"):
        QG.linear_sequential.LinearSequential(
            input_dim=10,
            hidden_dims=None,
            output_dims=[2, 3],
            activation=torch.nn.ReLU,
            backbone_kwargs=None,
            output_kwargs=[{}, {}],
        )


def test_linseq_empty_hidden_dims():
    linseq = QG.linear_sequential.LinearSequential(
        input_dim=10,
        hidden_dims=[],
        output_dims=[2, 3],
        activation=torch.nn.ReLU,
        backbone_kwargs=None,
        output_kwargs=[{}, {}],
    )

    assert isinstance(linseq.backbone, torch.nn.Identity)
    assert len(linseq.output_layers) == 2
    assert linseq.output_layers[0].in_channels == 10
    assert linseq.output_layers[0].out_channels == 2
    assert linseq.output_layers[1].in_channels == 10
    assert linseq.output_layers[1].out_channels == 3


def test_linseq_invalid_hidden_dims(linseq_params):
    with pytest.raises(
        ValueError, match="hidden_dims must be a list of positive integers"
    ):
        QG.linear_sequential.LinearSequential(
            input_dim=linseq_params["input_dim"],
            hidden_dims=[-1, 3],
            output_dims=linseq_params["output_dims"],
            activation=linseq_params["activation"],
            backbone_kwargs=linseq_params["backbone_kwargs"],
            output_kwargs=linseq_params["output_kwargs"],
        )


def test_linseq_invalid_output_dims(linseq_params):
    with pytest.raises(
        ValueError, match="output_dims must be a non-empty list of integers"
    ):
        QG.linear_sequential.LinearSequential(
            input_dim=linseq_params["input_dim"],
            hidden_dims=linseq_params["hidden_dims"],
            output_dims=[],
            activation=linseq_params["activation"],
            backbone_kwargs=linseq_params["backbone_kwargs"],
            output_kwargs=linseq_params["output_kwargs"],
        )
    with pytest.raises(
        ValueError, match="output_dims must be a list of positive integers"
    ):
        QG.linear_sequential.LinearSequential(
            input_dim=linseq_params["input_dim"],
            hidden_dims=linseq_params["hidden_dims"],
            output_dims=linseq_params["output_dims"] + [-1],
            activation=linseq_params["activation"],
            backbone_kwargs=linseq_params["backbone_kwargs"],
            output_kwargs=linseq_params["output_kwargs"],
        )


def test_linseq_forward_pass_with_kwargs(linseq_params):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    backbone_kwargs = [{"bias": True}, {"bias": False}]
    output_kwargs = [{"bias": True}, {"bias": True}]
    activation_kwargs = [{"inplace": False}, {"inplace": False}]

    cl = QG.linear_sequential.LinearSequential(
        input_dim=linseq_params["input_dim"],
        hidden_dims=linseq_params["hidden_dims"],
        output_dims=linseq_params["output_dims"],
        activation=linseq_params["activation"],
        backbone_kwargs=backbone_kwargs,
        output_kwargs=output_kwargs,
        activation_kwargs=activation_kwargs,
    )

    outputs = cl(input_tensor)

    assert len(outputs) == 2  # Two output layers
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs[0].shape == (5, 2)  # First output layer shape
    assert outputs[1].shape == (5, 3)  # Second output layer shape


def test_linseq_forward_pass(linearseq):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    outputs = linearseq(input_tensor)

    assert len(outputs) == 2  # Two output layers
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs[0].shape == (5, 2)  # First output layer shape
    assert outputs[1].shape == (5, 3)  # Second output layer shape


def test_linseq_backward_pass(linearseq):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10

    linearseq.train()  # Set the model to training mode
    outputs = linearseq(input_tensor)

    loss = sum(output.mean() for output in outputs)
    loss.backward()
    assert linearseq.backbone[0].weight.grad is not None
    assert linearseq.backbone[2].weight.grad is not None
    assert linearseq.output_layers[0].weight.grad is not None
    assert linearseq.output_layers[1].weight.grad is not None
    assert linearseq.backbone[0].bias.grad is not None
    assert linearseq.backbone[2].bias.grad is not None
    assert linearseq.output_layers[0].bias.grad is not None
    assert linearseq.output_layers[1].bias.grad is not None


def test_linseq_from_config():
    "test construction of a linearsequential instance from config dict"
    config = {
        "input_dim": 10,
        "hidden_dims": [20, 30],
        "output_dims": [2, 3],
        "activation": "relu",
        "backbone_kwargs": [{}, {}],
        "output_kwargs": [{}, {}],
    }

    linearseq = QG.linear_sequential.LinearSequential.from_config(config)

    assert isinstance(linearseq, QG.linear_sequential.LinearSequential)
    assert len(linearseq.backbone) == 4
    assert len(linearseq.output_layers) == 2
    assert linearseq.backbone[0].in_channels == 10
    assert linearseq.backbone[0].out_channels == 20
    assert linearseq.backbone[2].in_channels == 20
    assert linearseq.backbone[2].out_channels == 30
    assert linearseq.output_layers[0].in_channels == 30
    assert linearseq.output_layers[0].out_channels == 2
    assert linearseq.output_layers[1].in_channels == 30
    assert linearseq.output_layers[1].out_channels == 3


def test_linseq_save_load(linearseq, tmp_path):
    "test saving of the linearseq model"
    linearseq.save(tmp_path / "model.pt")

    assert (tmp_path / "model.pt").exists()

    loaded_linseq = QG.LinearSequential.load(tmp_path / "model.pt")

    assert loaded_linseq.state_dict() == linearseq.state_dict()
