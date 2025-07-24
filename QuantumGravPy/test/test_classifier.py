import QuantumGrav as QG
import pytest
import torch
import torch_geometric


@pytest.fixture
def classifier_params():
    return {
        "input_dim": 10,
        "hidden_dims": [20, 30],
        "output_dims": [2, 3],
        "activation": torch.nn.ReLU,
        "backbone_kwargs": [{}, {}],
        "output_kwargs": [
            {},
        ],
        "activation_kwargs": [
            {"inplace": False},
        ],
    }


@pytest.fixture
def classifier_config(classifier_params):
    return {
        "input_dim": classifier_params["input_dim"],
        "output_dims": classifier_params["output_dims"],
        "hidden_dims": classifier_params["hidden_dims"],
        "activation": "relu",
        "backbone_kwargs": classifier_params["backbone_kwargs"],
        "output_kwargs": classifier_params["output_kwargs"],
        "activation_kwargs": classifier_params["activation_kwargs"],
    }


def test_classifier_block_creation(classifier_params):
    cblock = QG.ClassifierBlock(
        input_dim=classifier_params["input_dim"],
        output_dims=classifier_params["output_dims"],
        hidden_dims=classifier_params["hidden_dims"],
        activation=classifier_params["activation"],
        backbone_kwargs=classifier_params["backbone_kwargs"],
        activation_kwargs=classifier_params["activation_kwargs"],
        output_kwargs=classifier_params["output_kwargs"],
    )

    assert len(cblock.backbone) == 4  # 2 hidden layers + 2 activations
    assert isinstance(cblock.backbone[0], torch_geometric.nn.dense.Linear)
    assert isinstance(cblock.backbone[1], torch.nn.ReLU)
    assert isinstance(cblock.backbone[2], torch_geometric.nn.dense.Linear)
    assert isinstance(cblock.backbone[3], torch.nn.ReLU)
    assert cblock.backbone[0].in_channels == 10
    assert cblock.backbone[0].out_channels == 20
    assert cblock.backbone[2].in_channels == 20
    assert cblock.backbone[2].out_channels == 30
    assert cblock.output_layers[0].in_channels == 30
    assert cblock.output_layers[0].out_channels == 2


def test_classifier_block_from_config(classifier_config):
    cblock = QG.ClassifierBlock.from_config(classifier_config)

    assert len(cblock.backbone) == 4  # 2 hidden layers + 2 activations
    assert isinstance(cblock.backbone[0], torch_geometric.nn.dense.Linear)
    assert isinstance(cblock.backbone[1], torch.nn.ReLU)
    assert isinstance(cblock.backbone[2], torch_geometric.nn.dense.Linear)
    assert isinstance(cblock.backbone[3], torch.nn.ReLU)
    assert cblock.backbone[0].in_channels == 10
    assert cblock.backbone[0].out_channels == 20
    assert cblock.backbone[2].in_channels == 20
    assert cblock.backbone[2].out_channels == 30
    assert cblock.output_layers[0].in_channels == 30
    assert cblock.output_layers[0].out_channels == 2
