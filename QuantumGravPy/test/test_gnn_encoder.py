import QuantumGrav as QG
import torch
import pytest
from functools import partial
import copy


def cat_graph_features(*features, dim=1):
    return torch.cat(features, dim=dim)


QG.utils.register_graph_features_aggregation("cat_graph_features", cat_graph_features)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def encoder(gnn_block, pooling_layer):
    return QG.EncoderModule(
        encoder=[gnn_block,],
        pooling_layers=[pooling_layer, pooling_layer],
        aggregate_pooling=torch.cat,
    )


@pytest.fixture
def graph_features_net():
    return QG.LinearSequential(
        dims=[(10, 32), (32, 24), (24, 32)],
        activations=[torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        linear_kwargs=[{"bias": True}, {"bias": True}, {"bias": False}],
        activation_kwargs=[{"inplace": False}, {"inplace": False}, {"inplace": False}],
    )


@pytest.fixture
def encoder_with_graph_features(
    encoder_block, pooling_layer, graph_features_net
):
    return QG.EncoderModule(
        encoder=[encoder_block],
        pooling_layers=[pooling_layer],
        graph_features_net=graph_features_net,
        aggregate_graph_features=cat_graph_features,
        aggregate_pooling=torch.cat,
    )


@pytest.fixture
def encoder_with_graph_features_no_pooling(
    encoder_block, graph_features_net
):
    return QG.EncoderModule(
        encoder=[encoder_block],
        graph_features_net=graph_features_net,
        aggregate_graph_features=partial(cat_graph_features, dim=0),
    )


@pytest.fixture
def encoder_config():
    # direct analogue of gnn_model_config but only encoder relevant entries
    return {
        "encoder": [
            {
                "in_dim": 16,
                "out_dim": 32,
                "dropout": 0.3,
                "gnn_layer_type": "gcn",
                "normalizer": "batch_norm",
                "activation": "relu",
                "norm_args": [32],
                "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
                "projection_args": [16, 32],
                "projection_kwargs": {"bias": False},
                "gnn_layer_kwargs": {
                    "cached": False,
                    "bias": True,
                    "add_self_loops": True,
                },
            },
            {
                "in_dim": 32,
                "out_dim": 16,
                "dropout": 0.3,
                "gnn_layer_type": "gcn",
                "normalizer": "batch_norm",
                "activation": "relu",
                "norm_args": [16],
                "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
                "projection_args": [32, 16],
                "projection_kwargs": {"bias": False},
                "gnn_layer_kwargs": {
                    "cached": False,
                    "bias": True,
                    "add_self_loops": True,
                },
            },
        ],
        "pooling_layers": [
            {"type": "mean", "args": [], "kwargs": {}},
        ],
        "aggregate_pooling": {"type": "cat1", "args": [], "kwargs": {}},
        "aggregate_graph_features": {"type": "cat1", "args": [], "kwargs": {}},
        "graph_features_net": {
            "dims": [(10, 24), (24, 8), (8, 32)],
            "activations": ["relu", "relu", "sigmoid"],
            "linear_kwargs": [{}, {}, {}],
            "activation_kwargs": [{"inplace": False}, {"inplace": False}, {}],
        },
    }


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_encoder_creation(encoder):
    assert isinstance(encoder.encoder, torch.nn.ModuleList)
    assert len(encoder.encoder) == 1
    assert isinstance(encoder.encoder[0], QG.GNNBlock)

    assert isinstance(encoder.pooling_layers[0], QG.gnn_autoencoder.ModuleWrapper)
    assert isinstance(encoder.pooling_layers[1], QG.gnn_autoencoder.ModuleWrapper)
    assert isinstance(encoder.aggregate_pooling, QG.gnn_autoencoder.ModuleWrapper)

    assert encoder.encoder[0].in_features == 16
    assert encoder.encoder[-1].out_features == 32


def test_encoder_creation_pooling_inconsistent(encoder_config):
    encoder_config["pooling_layers"] = None
    with pytest.raises(ValueError):
        QG.EncoderModule.from_config(encoder_config)


def test_encoder_creation_no_pooling(encoder_config):
    encoder_config["pooling_layers"] = None
    encoder_config["aggregate_pooling"] = None
    enc = QG.EncoderModule.from_config(encoder_config)

    assert isinstance(enc.encoder, torch.nn.ModuleList)
    assert len(enc.encoder) == 2
    assert isinstance(enc.encoder[0], QG.GNNBlock)
    assert isinstance(enc.encoder[1], QG.GNNBlock)

    assert enc.pooling_layers is None
    assert enc.aggregate_pooling is None


def test_encoder_get_embeddings(encoder):
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    encoder.eval()
    out = encoder.get_embeddings(x, edge_index, batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 32 * 2)

    out_single = encoder.get_embeddings(x, edge_index)
    assert isinstance(out_single, torch.Tensor)
    assert out_single.shape == (5, 32)  # no pooling → node embeddings


def test_encoder_forward(encoder):
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    encoder.eval()
    out = encoder(x, edge_index, batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 32 * 2)


def test_encoder_forward_with_graph_features(encoder_with_graph_features):
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    gf = torch.randn(2, 10)

    encoder_with_graph_features.eval()
    out = encoder_with_graph_features(x, edge_index, batch, graph_features=gf)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 32 + 32)  # structural 32 + graph_feature 32


def test_encoder_forward_without_pooling(encoder_with_graph_features_no_pooling):
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    gf = torch.randn(2, 10)

    assert encoder_with_graph_features_no_pooling.pooling_layers is None
    assert encoder_with_graph_features_no_pooling.aggregate_pooling is None

    out = encoder_with_graph_features_no_pooling(
        x, edge_index, batch, graph_features=gf
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 32 + 32)  # node-level embedding + graph feature embedding


def test_encoder_to_config(encoder_with_graph_features):
    cfg = encoder_with_graph_features.to_config()
    assert "encoder" in cfg
    assert "pooling_layers" in cfg
    assert "graph_features_net" in cfg
    assert "aggregate_graph_features" in cfg
    assert "aggregate_pooling" in cfg


def test_encoder_from_config(encoder_config):
    enc = QG.EncoderModule.from_config(encoder_config)

    assert isinstance(enc.encoder, torch.nn.ModuleList)
    assert len(enc.encoder) == 2
    
    assert enc.pooling_layers is not None
    for pooling in model.pooling_layers:
        assert isinstance(pooling, QG.gnn_autoencoder.ModuleWrapper)
    assert isinstance(enc.pooling_layers, torch.nn.ModuleList)
    assert isinstance(enc.graph_features_net, QG.LinearSequential)

    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])

    enc.eval()
    out = enc(x, edge_index, batch)
    assert isinstance(out, torch.Tensor)


def test_encoder_repeated_forward_stability(encoder):
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1], [1, 3]])
    batch = torch.tensor([0, 0, 0, 1, 1])

    encoder.eval()
    out1 = encoder(x, edge_index, batch)
    out2 = encoder(x, edge_index, batch)

    assert torch.allclose(out1, out2, atol=1e-6)