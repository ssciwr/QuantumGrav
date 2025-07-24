import QuantumGrav as QG
import pytest
import torch
import torch_geometric


@pytest.fixture
def gnn_block():
    return QG.GNNBlock(
        in_channels=16,
        out_channels=32,
        dropout=0.3,
        gnn_layer_type=torch_geometric.nn.conv.GCNConv,
        normalizer=torch.nn.BatchNorm1d,
        activation=torch.nn.ReLU,
        gnn_layer_args=[],
        gnn_layer_kwargs={"cached": False, "bias": True, "add_self_loops": True},
        norm_args=[
            32,
        ],
        norm_kwargs={"eps": 1e-5, "momentum": 0.2},
    )


@pytest.fixture
def classifier_block():
    return QG.ClassifierBlock(
        input_dim=10,
        hidden_dims=[20, 30],
        output_dims=[2, 3],
        activation=torch.nn.ReLU,
        backbone_kwargs=[{}, {}],
        activation_kwargs=[{"inplace": False}],
        output_kwargs=[
            {},
        ],
    )


@pytest.fixture
def pooling_layer():
    return torch_geometric.nn.global_mean_pool


@pytest.fixture
def graph_features_net():
    return QG.GraphFeaturesBlock(
        input_dim=10,
        output_dim=[2, 3],
        hidden_dims=[20, 30],
        activation=torch.nn.ReLU,
        layer_kwargs=[{}, {}],
        activation_kwargs=[
            {"inplace": False},
        ],
    )


@pytest.fixture
def gnn_model_config():
    config = {
        "gcn_net": {
            "in_channels": 16,
            "out_channels": 32,
            "dropout": 0.3,
            "gnn_layer_type": "gcn",
            "normalizer": "batch_norm",
            "activation": "relu",
            "norm_args": [
                32,
            ],
            "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
            "gnn_layer_kwargs": {"cached": False, "bias": True, "add_self_loops": True},
        },
        "classifier": {
            "input_dim": 10,
            "output_dims": [2, 3],
            "hidden_dims": [20, 30],
            "activation": "relu",
            "backbone_kwargs": [{}, {}],
            "output_kwargs": [{}],
            "activation_kwargs": [{"inplace": False}],
        },
        "pooling_layer": "mean",
        "graph_features_net": {
            "input_dim": 10,
            "hidden_dims": [20, 30],
            "output_dims": [2, 3],
            "activation": "relu",
            "layer_kwargs": [{}, {}],
            "activation_kwargs": [
                {"inplace": False},
            ],
        },
    }
    return config


def test_gnn_model_creation(
    gnn_block, classifier_block, pooling_layer, graph_features_net
):
    """Test the creation of GNNModel with required components."""
    model = QG.GNNModel(
        gcn_net=gnn_block,
        classifier=classifier_block,
        pooling_layer=pooling_layer,
        graph_features_net=graph_features_net,
    )

    assert isinstance(model.gcn_net, QG.GNNBlock)
    assert isinstance(model.classifier, QG.ClassifierBlock)
    assert isinstance(model.pooling_layer, QG.PoolingLayer)
    assert isinstance(model.graph_features_net, QG.GraphFeaturesNet)


# def test_gnn_model_forward(
#     gnn_model_config, gnn_block, classifier_block, pooling_layer, graph_features_net
# ):
#     pass


# def test_gnn_model_get_embeddings(
#     gnn_model_config, gnn_block, classifier_block, pooling_layer, graph_features_net
# ):
#     pass


# def test_gnn_model_forward_with_graph_features(
#     gnn_model_config, gnn_block, classifier_block, pooling_layer, graph_features_net
# ):
#     pass


def test_gnn_model_creation_from_config(gnn_model_config):
    model = QG.GNNModel(**gnn_model_config)
    assert isinstance(model.gcn_net, QG.GNNBlock)
    assert isinstance(model.classifier, QG.ClassifierBlock)
    assert isinstance(model.pooling_layer, QG.PoolingLayer)
    assert isinstance(model.graph_features_net, QG.GraphFeaturesNet)
