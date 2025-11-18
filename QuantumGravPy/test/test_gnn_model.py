import QuantumGrav as QG
import torch
import torch_geometric
import pytest
from jsonschema import ValidationError


def test_gnn_model_creation(gnn_model):
    """Test the creation of GNNModel with required components."""

    assert isinstance(gnn_model.encoder, QG.SequentialModel)
    assert len(gnn_model.encoder.layers) == 4
    assert isinstance(gnn_model.downstream_tasks["0"], torch_geometric.nn.dense.Linear)
    assert isinstance(gnn_model.downstream_tasks["1"], torch_geometric.nn.dense.Linear)

    assert isinstance(gnn_model.pooling_layers[0], QG.gnn_model.ModuleWrapper)
    assert isinstance(gnn_model.pooling_layers[1], QG.gnn_model.ModuleWrapper)
    assert isinstance(gnn_model.aggregate_pooling, QG.gnn_model.ModuleWrapper)
    assert len(gnn_model.downstream_tasks) == 2


def test_gnn_model_creation_pooling_aggregations_inconsistent(model_config):
    """Test the creation of GNNModel with missing aggregation function."""

    model_config["pooling_layers"] = None

    with pytest.raises(ValidationError):
        QG.GNNModel.from_config(model_config)


def test_gnn_model_get_embeddings(gnn_model):
    """Test the get_embeddings method of GNNModel."""

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model.get_graph_embeddings(x, edge_index, None, batch)
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 32)  # concatenated over pooling layers

    output_single = gnn_model.get_graph_embeddings(x, edge_index, None, None)  # no batch -> single graph
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
    "test the model forward pass with active/inactive tasks"
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

    gnn_model.set_task_inactive(0)
    assert gnn_model.active_tasks == [False, True]
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 1
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes


@pytest.mark.parametrize("to_disable", [1, 0])
def test_gnn_model_forward_set_active_backprop(gnn_model, to_disable):
    "test the model forward pass with active/inactive tasks and backpropagation"
    task_to_keep = 0 if to_disable == 1 else 1

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.train()  # Set model to training mode

    y = torch.randn(4, 3)  # Random target values
    compute_loss = torch.nn.MSELoss()

    # baseline - both tasks active
    assert gnn_model.training is True  # Ensure model is in train mode
    assert gnn_model.active_tasks == [True, True]
    output = gnn_model(x, edge_index, batch)
    assert 0 in output and 1 in output
    loss = sum(
        [
            compute_loss(output[i], y)
            for i in range(len(gnn_model.active_tasks))
            if gnn_model.active_tasks[i]
        ]
    )

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.named_parameters():
        assert p.grad is not None

    assert len(output) == 2
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 2 classes

    # reset grads
    for p in gnn_model.parameters():
        p.grad = None

    # disable task 'to_disable'
    gnn_model.set_task_inactive(to_disable)
    assert gnn_model.active_tasks[to_disable] is False
    assert gnn_model.active_tasks[task_to_keep] is True
    output = gnn_model(x, edge_index, batch)
    loss = sum(
        [
            compute_loss(output[i], y)
            for i in range(len(gnn_model.active_tasks))
            if gnn_model.active_tasks[i]
        ]
    )

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.encoder.named_parameters():
        assert p.grad is not None

    for _, p in gnn_model.downstream_tasks[task_to_keep].named_parameters():
        assert p.grad is not None

    for _, p in gnn_model.downstream_tasks[to_disable].named_parameters():
        assert p.grad is None

    assert len(output) == 1
    assert output[task_to_keep].shape == (
        4,
        3,
    )  # 2 graphs, 2 concat pooling layers, 3 classes
    # reset grads
    for p in gnn_model.parameters():
        p.grad = None

    # re-enable task 'to_disable'
    gnn_model.set_task_active(to_disable)
    assert gnn_model.active_tasks == [True, True]
    output = gnn_model(x, edge_index, batch)
    loss = sum(
        [
            compute_loss(output[i], y)
            for i in range(len(gnn_model.active_tasks))
            if gnn_model.active_tasks[i]
        ]
    )

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.named_parameters():
        assert p.grad is not None

    assert len(output) == 2
    assert output[0].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (4, 3)  # 2 graphs, 2 concat pooling layers, 2 classes
    # reset grads
    for p in gnn_model.parameters():
        p.grad = None


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


def test_gnn_model_to_config(gnn_model_with_graph_features):
    cfg = gnn_model_with_graph_features.to_config()
    assert "encoder" in cfg
    assert "downstream_tasks" in cfg
    assert "pooling_layers" in cfg
    assert "graph_features_net" in cfg
    assert "aggregate_graph_features" in cfg


def test_gnn_model_creation_from_config(gnn_model_config):
    "test gnn model initialization from config file"
    model = QG.GNNModel.from_config(gnn_model_config)

    assert model.active_tasks == [False, True]
    assert isinstance(model.encoder, torch.nn.ModuleList)

    assert len(model.encoder) == 2  # Assuming two GNN blocks

    for task in model.downstream_tasks:
        assert isinstance(task, QG.LinearSequential)

    assert model.pooling_layers is not None

    for pooling in model.pooling_layers:
        assert isinstance(pooling, QG.gnn_model.ModuleWrapper)

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

    gnn_model_with_graph_features.eval()
    loaded_gnn_model.eval()
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 10)  # 2 graphs with 10 features each

    y = gnn_model_with_graph_features.forward(
        x, edge_index, batch, graph_features=graph_features
    )

    y_loaded = loaded_gnn_model.forward(
        x, edge_index, batch, graph_features=graph_features
    )
    for i in y.keys():
        assert torch.allclose(y[i], y_loaded[i], atol=1e-8)
