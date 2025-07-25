import QuantumGrav as QG
import torch
import torch_geometric


def test_gnn_model_creation(gnn_model):
    """Test the creation of GNNModel with required components."""

    assert isinstance(gnn_model.gcn_net, QG.GNNBlock)
    assert isinstance(gnn_model.classifier, QG.ClassifierBlock)
    assert gnn_model.pooling_layer == torch_geometric.nn.global_mean_pool
    assert isinstance(gnn_model.graph_features_net, QG.GraphFeaturesBlock)

    # assert sizes and properties of the components
    assert gnn_model.gcn_net.in_dim == 16
    assert gnn_model.gcn_net.out_dim == 32
    assert gnn_model.classifier.output_layers[0].in_channels == 12
    assert gnn_model.classifier.output_layers[1].in_channels == 12
    assert gnn_model.classifier.output_layers[0].out_channels == 2
    assert gnn_model.classifier.output_layers[1].out_channels == 3


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
    assert output.shape == (2, 32)  # pooled across nodes -> two graphs

    output_single = gnn_model.get_embeddings(x, edge_index)  # no batch -> single graph
    assert isinstance(output_single, torch.Tensor)
    assert output_single.shape == (1, 32)  # pooled across nodes -> single graph


def test_gnn_model_forward(gnn_model):
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model(x, edge_index, batch)
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, list)
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs, 3 classes


def test_gnn_model_forward_with_graph_features(gnn_model_with_graph_features):
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
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)


def test_gnn_model_creation_from_config(gnn_model_config):
    model = QG.GNNModel.from_config(gnn_model_config)
    assert isinstance(model.gcn_net, QG.GNNBlock)
    assert isinstance(model.classifier, QG.ClassifierBlock)
    assert model.pooling_layer == torch_geometric.nn.global_mean_pool
    assert isinstance(model.graph_features_net, QG.GraphFeaturesBlock)
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    model.eval()  # Set model to evaluation mode
    output = model(x, edge_index, batch)
    assert model.training is False  # Ensure model is in eval mode
    assert isinstance(output, list)
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs, 3 classes
