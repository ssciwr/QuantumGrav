import QuantumGrav as QG
import pytest
import torch
import torch_geometric


@pytest.fixture
def g_params():
    return {
        "input_dim": 10,
        "hidden_dims": [20, 30],
        "output_dims": 2,
        "activation": torch.nn.ReLU,
        "layer_kwargs": [{}, {}],
        "activation_kwargs": [
            {"inplace": False},
        ],
    }


@pytest.fixture
def g_block_config(g_params):
    return {
        "input_dim": g_params["input_dim"],
        "output_dim": g_params["output_dims"],
        "hidden_dims": g_params["hidden_dims"],
        "activation": "relu",
        "layer_kwargs": g_params["layer_kwargs"],
        "activation_kwargs": g_params["activation_kwargs"],
    }


def test_graph_features_block_creation(g_params):
    gblock = QG.GraphFeaturesBlock(
        input_dim=g_params["input_dim"],
        output_dim=g_params["output_dims"],
        hidden_dims=g_params["hidden_dims"],
        activation=g_params["activation"],
        layer_kwargs=g_params["layer_kwargs"],
        activation_kwargs=g_params["activation_kwargs"],
    )

    assert len(gblock.backbone) == 4  # 2 hidden layers + 2 activations
    assert isinstance(gblock.backbone[0], torch_geometric.nn.dense.Linear)
    assert isinstance(gblock.backbone[1], torch.nn.ReLU)
    assert isinstance(gblock.backbone[2], torch_geometric.nn.dense.Linear)
    assert isinstance(gblock.backbone[3], torch.nn.ReLU)
    assert gblock.backbone[0].in_channels == 10
    assert gblock.backbone[0].out_channels == 20
    assert gblock.backbone[2].in_channels == 20
    assert gblock.backbone[2].out_channels == 30
    assert gblock.output_layers[0].in_channels == 30
    assert gblock.output_layers[0].out_channels == 2


def test_graph_features_block_from_config(g_block_config):
    gblock = QG.GraphFeaturesBlock.from_config(g_block_config)

    assert len(gblock.backbone) == 4  # 2 hidden layers + 2 activations
    assert isinstance(gblock.backbone[0], torch_geometric.nn.dense.Linear)
    assert isinstance(gblock.backbone[1], torch.nn.ReLU)
    assert isinstance(gblock.backbone[2], torch_geometric.nn.dense.Linear)
    assert isinstance(gblock.backbone[3], torch.nn.ReLU)
    assert gblock.backbone[0].in_channels == 10
    assert gblock.backbone[0].out_channels == 20
    assert gblock.backbone[2].in_channels == 20
    assert gblock.backbone[2].out_channels == 30
    assert gblock.output_layers[0].in_channels == 30
    assert gblock.output_layers[0].out_channels == 2
