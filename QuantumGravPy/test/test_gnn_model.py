import QuantumGrav as QG
import torch
import torch_geometric
import pytest


def cat_graph_features(*features):
    return torch.cat(features, dim=1)


@pytest.fixture
def gnn_model(gnn_block, classifier_block, pooling_layer, graph_features_net):
    return QG.GNNModel(
        encoder=[
            gnn_block,
        ],
        downstream_tasks=[classifier_block, classifier_block],
        pooling_layers=[pooling_layer, pooling_layer],
        graph_features_net=graph_features_net,
        aggregate_pooling=torch.cat,
    )


@pytest.fixture
def gnn_model_with_graph_features(
    gnn_block, classifier_block_graphfeatures, pooling_layer, graph_features_net
):
    return QG.GNNModel(
        encoder=[
            gnn_block,
        ],
        downstream_tasks=[
            classifier_block_graphfeatures,
            classifier_block_graphfeatures,
        ],
        pooling_layers=[pooling_layer],
        aggregate_graph_features=cat_graph_features,
        graph_features_net=graph_features_net,
    )


@pytest.fixture
def gnn_model_config():
    config = {
        "encoder": [
            {
                "in_dim": 16,
                "out_dim": 32,
                "dropout": 0.3,
                "gnn_layer_type": "gcn",
                "normalizer": "batch_norm",
                "activation": "relu",
                "norm_args": [
                    32,
                ],
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
                "norm_args": [
                    16,
                ],
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
        "downstream_tasks": [
            {
                "input_dim": 16,
                "output_dims": [2, 3],
                "hidden_dims": [24, 18],
                "activation": "relu",
                "backbone_kwargs": [{}, {}],
                "output_kwargs": [{}],
                "activation_kwargs": [{"inplace": False}],
            },
        ],
        "pooling_layers": ["mean"],
        "aggregate_pooling": "test_registered",
        "aggregate_graph_features": "test_registered",
        "graph_features_net": {
            "input_dim": 10,
            "hidden_dims": [24, 8],
            "output_dim": 32,
            "activation": "relu",
            "layer_kwargs": [{}, {}],
            "activation_kwargs": [
                {"inplace": False},
            ],
        },
    }
    return config


def test_gnn_model_creation(gnn_model):
    """Test the creation of GNNModel with required components."""

    assert isinstance(gnn_model.encoder, torch.nn.ModuleList)
    assert len(gnn_model.encoder) == 1  # Assuming one GNN block
    assert isinstance(gnn_model.encoder[0], QG.GNNBlock)
    assert isinstance(gnn_model.downstream_tasks[0], QG.LinearSequential)
    assert isinstance(gnn_model.downstream_tasks[1], QG.LinearSequential)

    assert gnn_model.pooling_layers[0] == torch_geometric.nn.global_mean_pool
    assert gnn_model.pooling_layers[1] == torch_geometric.nn.global_mean_pool
    assert gnn_model.aggregate_pooling == torch.cat
    assert isinstance(gnn_model.graph_features_net, QG.LinearSequential)

    # assert sizes and properties of the components
    assert gnn_model.encoder[0].in_dim == 16
    assert gnn_model.encoder[0].out_dim == 32
    assert gnn_model.downstream_tasks[0].output_layers[0].in_channels == 12
    assert gnn_model.downstream_tasks[0].output_layers[1].in_channels == 12
    assert gnn_model.downstream_tasks[0].output_layers[0].out_channels == 2
    assert gnn_model.downstream_tasks[0].output_layers[1].out_channels == 3

    assert gnn_model.downstream_tasks[1].output_layers[0].in_channels == 12
    assert gnn_model.downstream_tasks[1].output_layers[1].in_channels == 12
    assert gnn_model.downstream_tasks[1].output_layers[0].out_channels == 2
    assert gnn_model.downstream_tasks[1].output_layers[1].out_channels == 3


def test_gnn_model_creation_pooling_aggregation_missing(gnn_model_config):
    """Test the creation of GNNModel with missing aggregation function."""

    QG.register_pooling_aggregation("test_registered", torch.cat)

    QG.register_graph_features_aggregation("test_registered", torch.cat)
    gnn_model_config["pooling_layers"].append("max")

    model = QG.GNNModel.from_config(gnn_model_config)

    assert model.pooling_layers == [
        torch_geometric.nn.global_mean_pool,
        torch_geometric.nn.global_max_pool,
    ]

    assert model.aggregate_graph_features == torch.cat


def test_gnn_model_get_embeddings(gnn_model):
    """Test the get_embeddings method of GNNModel."""

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model.get_embeddings(x, edge_index, batch)
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 32)  # concatenated over pooling layers

    output_single = gnn_model.get_embeddings(x, edge_index)  # no batch -> single graph
    assert isinstance(output_single, torch.Tensor)
    assert output_single.shape == (2, 32)  # concatenated over pooling layers


def test_gnn_model_forward(gnn_model):
    "test gnn model forward run"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model(x, edge_index, batch)

    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, list)
    assert output[0][0].shape == (4, 2)  # 2 graphs, 2 concat pooling layers, 2 classes
    assert output[0][1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1][0].shape == (4, 2)  # 2 graphs, 2 concat pooling layers, 2 classes
    assert output[1][1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes


def test_gnn_model_forward_with_graph_features(gnn_model_with_graph_features):
    "test gnn model forward run with graph features"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 10)  # 2 graphs with 10 features each
    gnn_model_with_graph_features.eval()  # Set model to evaluation mode
    output = gnn_model_with_graph_features(
        x, edge_index, batch, graph_features=graph_features
    )
    assert (
        gnn_model_with_graph_features.training is False
    )  # Ensure model is in eval mode
    assert isinstance(output, list)
    assert output[0][0].shape == (2, 2)  # 2 graphs, 1 pooling layers, 2 classes
    assert output[0][1].shape == (2, 3)


def test_gnn_model_creation_from_config(gnn_model_config):
    "test gnn model initialization from config file"
    model = QG.GNNModel.from_config(gnn_model_config)

    assert isinstance(model.encoder, torch.nn.ModuleList)

    assert len(model.encoder) == 2  # Assuming two GNN blocks

    for task in model.downstream_tasks:
        assert isinstance(task, QG.LinearSequential)

    assert model.pooling_layers == [torch_geometric.nn.global_mean_pool]

    assert isinstance(model.graph_features_net, QG.LinearSequential)

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    model.eval()  # Set model to evaluation mode
    output = model(x, edge_index, batch)
    assert model.training is False  # Ensure model is in eval mode
    assert isinstance(output, list)
    assert output[0][0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[0][1].shape == (2, 3)  # 2 graphs, 3 classes


def test_gnn_model_save_load(gnn_model_with_graph_features, tmp_path):
    "test saving and loading of the combined model"
    gnn_model_with_graph_features.save(tmp_path / "model.pt")

    assert (tmp_path / "model.pt").exists()

    loaded_gnn_model = QG.GNNModel.load(tmp_path / "model.pt")

    assert gnn_model_with_graph_features.graph_features_net is not None
    assert len(loaded_gnn_model.state_dict().keys()) == len(
        gnn_model_with_graph_features.state_dict().keys()
    )

    loaded_keys = set(loaded_gnn_model.state_dict().keys())
    original_keys = set(gnn_model_with_graph_features.state_dict().keys())

    assert loaded_keys == original_keys

    for k in loaded_gnn_model.state_dict().keys():
        assert torch.equal(
            loaded_gnn_model.state_dict()[k],
            gnn_model_with_graph_features.state_dict()[k],
        )
