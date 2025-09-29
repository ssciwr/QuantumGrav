import QuantumGrav as QG
import torch
import pytest
from functools import partial


def cat_graph_features(*features, dim=1):
    return torch.cat(features, dim=dim)


@pytest.fixture
def gnn_model(gnn_block, classifier_block, pooling_layer):
    return QG.GNNModel(
        encoder=[
            gnn_block,
        ],
        downstream_tasks=[classifier_block, classifier_block],
        pooling_layers=[pooling_layer, pooling_layer],
        aggregate_pooling=torch.cat,
        active_tasks=[True, True],
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
        aggregate_pooling=torch.cat,
        active_tasks=[True, True],
    )


@pytest.fixture
def classifier_block_graphfeatures_no_pooling():
    return QG.LinearSequential(
        input_dim=32,
        hidden_dims=[24, 12],
        output_dim=3,
        activation=torch.nn.ReLU,
        backbone_kwargs=[{}, {}],
        activation_kwargs=[{"inplace": False}],
        output_kwargs={},
    )


@pytest.fixture
def gnn_model_with_graph_features_no_pooling(
    gnn_block, classifier_block_graphfeatures_no_pooling, graph_features_net
):
    return QG.GNNModel(
        encoder=[
            gnn_block,
        ],
        downstream_tasks=[
            classifier_block_graphfeatures_no_pooling,
            classifier_block_graphfeatures_no_pooling,
        ],
        aggregate_graph_features=partial(cat_graph_features, dim=0),
        graph_features_net=graph_features_net,
        active_tasks=[True, True],
    )


@pytest.fixture
def gnn_model_config():
    config = {
        "active_tasks": [
            1,
        ],
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
                "output_dim": 3,
                "hidden_dims": [24, 18],
                "activation": "relu",
                "backbone_kwargs": [{}, {}],
                "output_kwargs": {},
                "activation_kwargs": [{"inplace": False}],
                "active": False,
            },
            {
                "input_dim": 16,
                "output_dim": 3,
                "hidden_dims": [24, 18],
                "activation": "relu",
                "backbone_kwargs": [{}, {}],
                "output_kwargs": {},
                "activation_kwargs": [{"inplace": False}],
                "active": True,
            },
        ],
        "pooling_layers": [
            {
                "type": "mean",
                "args": [],
                "kwargs": {},
            },
        ],
        "aggregate_pooling": {
            "type": "cat1",
            "args": [],
            "kwargs": {},
        },
        "aggregate_graph_features": {
            "type": "cat1",
            "args": [],
            "kwargs": {},
        },
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

    assert isinstance(gnn_model.pooling_layers[0], QG.gnn_model.PoolingWrapper)
    assert isinstance(gnn_model.pooling_layers[1], QG.gnn_model.PoolingWrapper)
    assert isinstance(gnn_model.aggregate_pooling, QG.gnn_model.PoolingWrapper)

    # assert sizes and properties of the components
    assert gnn_model.encoder[0].in_dim == 16
    assert gnn_model.encoder[0].out_dim == 32
    assert len(gnn_model.downstream_tasks) == 2


def test_gnn_model_creation_pooling_aggregations_inconsistent(gnn_model_config):
    """Test the creation of GNNModel with missing aggregation function."""

    gnn_model_config["pooling_layers"] = None

    with pytest.raises(ValueError):
        QG.GNNModel.from_config(gnn_model_config)


def test_gnn_model_creation_pooling_no_aggregations(gnn_model_config):
    """Test the creation of GNNModel with missing aggregation function."""

    gnn_model_config["pooling_layers"] = None
    gnn_model_config["aggregate_pooling"] = None

    gnn_model = QG.GNNModel.from_config(gnn_model_config)

    assert isinstance(gnn_model.encoder, torch.nn.ModuleList)
    assert len(gnn_model.encoder) == 2  # Assuming one GNN block
    assert isinstance(gnn_model.encoder[0], QG.GNNBlock)
    assert isinstance(gnn_model.encoder[1], QG.GNNBlock)
    assert isinstance(gnn_model.downstream_tasks[0], QG.LinearSequential)
    assert isinstance(gnn_model.downstream_tasks[1], QG.LinearSequential)

    assert gnn_model.pooling_layers is None
    assert gnn_model.aggregate_pooling is None


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
    assert isinstance(output, dict)
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 2 classes


def test_gnn_model_forward_set_active(gnn_model):
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    assert gnn_model.training is False  # Ensure model is in eval mode

    assert gnn_model.active_tasks == [True, True]
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 2
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 2 classes

    gnn_model.set_task_inactive(1)
    assert gnn_model.active_tasks == [True, False]
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 1
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes

    gnn_model.set_task_active(1)
    assert gnn_model.active_tasks == [True, True]
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 2
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 2 classes


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
    assert isinstance(output, dict)
    assert output[0].shape == (2, 3)  # 2 graphs, 1 pooling layers, 2 classes
    assert output[1].shape == (2, 3)


def test_gnn_model_forward_without_pooling(gnn_model_with_graph_features_no_pooling):
    "test gnn model without adding pooling"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 10)  # 2 graphs with 10 features each
    assert gnn_model_with_graph_features_no_pooling.pooling_layers is None
    assert gnn_model_with_graph_features_no_pooling.aggregate_pooling is None
    output = gnn_model_with_graph_features_no_pooling(
        x, edge_index, batch, graph_features=graph_features
    )

    assert output[0].shape == (7, 3)
    assert output[1].shape == (7, 3)


def test_gnn_model_creation_from_config(gnn_model_config):
    "test gnn model initialization from config file"
    model = QG.GNNModel.from_config(gnn_model_config)

    assert model.active_tasks == [False, True]
    assert isinstance(model.encoder, torch.nn.ModuleList)

    assert len(model.encoder) == 2  # Assuming two GNN blocks

    for task in model.downstream_tasks:
        assert isinstance(task, QG.LinearSequential)

    for pooling in model.pooling_layers:
        assert isinstance(pooling, QG.gnn_model.PoolingWrapper)

    assert isinstance(model.graph_features_net, QG.LinearSequential)

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    model.eval()  # Set model to evaluation mode
    output = model(x, edge_index, batch)
    assert model.training is False  # Ensure model is in eval mode
    assert isinstance(output, dict)
    assert 0 not in output
    assert output[1].shape == (2, 3)  # 2 graphs, 3 classes


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
