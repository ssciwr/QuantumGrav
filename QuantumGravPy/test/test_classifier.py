import pytest
import torch
import QuantumGrav as QG


@pytest.fixture
def classifier_params():
    return {
        "input_dim": 10,
        "hidden_dims": [20, 30],
        "output_dims": [2, 3],
        "activation": torch.nn.ReLU,
        "backbone_kwargs": [{}, {}],
        "output_kwargs": [{}, {}],
    }


@pytest.fixture
def classifier_block(classifier_params):
    params = classifier_params
    return QG.ClassifierBlock(
        input_dim=params["input_dim"],
        hidden_dims=params["hidden_dims"],
        output_dims=params["output_dims"],
        activation=params["activation"],
        backbone_kwargs=params["backbone_kwargs"],
        output_kwargs=params["output_kwargs"],
    )


def test_classifier_block_creation(classifier_params):
    classifier_block = QG.ClassifierBlock(
        input_dim=classifier_params["input_dim"],
        hidden_dims=classifier_params["hidden_dims"],
        output_dims=classifier_params["output_dims"],
        activation=classifier_params["activation"],
        backbone_kwargs=classifier_params["backbone_kwargs"],
        output_kwargs=classifier_params["output_kwargs"],
    )
    assert len(classifier_block.backbone) == 4  # 2 hidden layers + 2 activations
    assert len(classifier_block.output_layers) == 2
    assert isinstance(classifier_block.backbone[0], torch.nn.Linear)
    assert isinstance(classifier_block.backbone[1], torch.nn.ReLU)
    assert isinstance(classifier_block.backbone[2], torch.nn.Linear)
    assert isinstance(classifier_block.backbone[3], torch.nn.ReLU)
    assert isinstance(classifier_block.output_layers[0], torch.nn.Linear)
    assert isinstance(classifier_block.output_layers[1], torch.nn.Linear)
    assert classifier_block.backbone[0].in_features == 10
    assert classifier_block.backbone[0].out_features == 20
    assert classifier_block.backbone[2].in_features == 20
    assert classifier_block.backbone[2].out_features == 30
    assert classifier_block.output_layers[0].in_features == 30
    assert classifier_block.output_layers[0].out_features == 2
    assert classifier_block.output_layers[1].in_features == 30
    assert classifier_block.output_layers[1].out_features == 3


def test_classifier_block_invalid_hidden_dims(classifier_params):
    params = classifier_params
    params["hidden_dims"] = [-1, 3]  # Invalid hidden dimension
    with pytest.raises(
        ValueError, match="hidden_dims must be a list of positive integers"
    ):
        QG.ClassifierBlock(
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dims=params["output_dims"],
            activation=params["activation"],
            backbone_kwargs=params["backbone_kwargs"],
            output_kwargs=params["output_kwargs"],
        )

    params["hidden_dims"] = []  # Empty hidden dimensions
    with pytest.raises(
        ValueError, match="hidden_dims must be a non-empty list of integers"
    ):
        QG.ClassifierBlock(
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dims=params["output_dims"],
            activation=params["activation"],
            backbone_kwargs=params["backbone_kwargs"],
            output_kwargs=params["output_kwargs"],
        )


def test_classifier_block_invalid_output_dims(classifier_params):
    params = classifier_params
    params["output_dims"] = [-1, 3]  # Invalid output dimension
    with pytest.raises(
        ValueError, match="output_dims must be a list of positive integers"
    ):
        QG.ClassifierBlock(
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dims=params["output_dims"],
            activation=params["activation"],
            backbone_kwargs=params["backbone_kwargs"],
            output_kwargs=params["output_kwargs"],
        )

    params["output_dims"] = []  # Empty output dimensions
    with pytest.raises(
        ValueError, match="output_dims must be a non-empty list of integers"
    ):
        QG.ClassifierBlock(
            input_dim=params["input_dim"],
            hidden_dims=params["hidden_dims"],
            output_dims=params["output_dims"],
            activation=params["activation"],
            backbone_kwargs=params["backbone_kwargs"],
            output_kwargs=params["output_kwargs"],
        )


def test_classifier_block_forward_pass_with_kwargs(classifier_params):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    backbone_kwargs = [{"bias": True}, {"bias": False}]
    output_kwargs = [{"bias": True}, {"bias": True, "dtype": torch.float32}]
    activation_kwargs = [{"inplace": False}, {"inplace": False}]

    cl = QG.ClassifierBlock(
        input_dim=classifier_params["input_dim"],
        hidden_dims=classifier_params["hidden_dims"],
        output_dims=classifier_params["output_dims"],
        activation=classifier_params["activation"],
        backbone_kwargs=backbone_kwargs,
        output_kwargs=output_kwargs,
        activation_kwargs=activation_kwargs,
    )

    outputs = cl(input_tensor)

    assert len(outputs) == 2  # Two output layers
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs[0].shape == (5, 2)  # First output layer shape
    assert outputs[1].shape == (5, 3)  # Second output layer shape


def test_classifier_block_forward_pass(classifier_block):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10
    outputs = classifier_block(input_tensor)

    assert len(outputs) == 2  # Two output layers
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert outputs[0].shape == (5, 2)  # First output layer shape
    assert outputs[1].shape == (5, 3)  # Second output layer shape


def test_classifier_block_backward_pass(classifier_block):
    input_tensor = torch.randn(5, 10)  # Batch size of 5, input dimension of 10

    classifier_block.train()  # Set the model to training mode
    outputs = classifier_block(input_tensor)

    loss = sum(output.mean() for output in outputs)
    loss.backward()
    assert classifier_block.backbone[0].weight.grad is not None
    assert classifier_block.backbone[2].weight.grad is not None
    assert classifier_block.output_layers[0].weight.grad is not None
    assert classifier_block.output_layers[1].weight.grad is not None
    assert classifier_block.backbone[0].bias.grad is not None
    assert classifier_block.backbone[2].bias.grad is not None
    assert classifier_block.output_layers[0].bias.grad is not None
    assert classifier_block.output_layers[1].bias.grad is not None
