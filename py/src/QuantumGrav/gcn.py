import torch
from torch_geometric.nn.conv import GCNConv
from torch.nn import Linear
import torch.nn.functional as F


class GCNBlock(torch.nn.Module):
    """
    A single GCN block with skip connections and optional batch normalization.
    This block can be used in a GCN backbone for graph neural networks.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.5,
        gcn_type=GCNConv,
        normalizer=torch.nn.Identity,
        activation=F.relu,
        gcn_kwargs=None,
    ):
        """
        Initializes the GCNBlock. A single GCN block with skip connections and optional batch normalization.
        This block can be used in a GCN backbone for graph neural networks.
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            dropout (float): Dropout rate.
            gcn_type (torch_geometric.nn.conv.GCNConv): Type of GCN layer to use.
            normalizer (torch.nn.Module): Normalization layer, e.g., BatchNorm1d.
            activation (callable): Activation function to apply after the GCN layer.
            gcn_kwargs (dict, optional): Additional keyword arguments for the GCN layer.
        """

        super(GCNBlock, self).__init__()
        self.dropout = dropout
        self.gcn_type = gcn_type
        self.conv = gcn_type(
            input_dim, output_dim, **(gcn_kwargs if gcn_kwargs else {})
        )
        self.activation = activation
        self.batch_norm = normalizer

        if input_dim != output_dim:
            # Use 1x1 convolution for projection
            self.projection = Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = torch.nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, kwargs=None) -> torch.Tensor:
        """
        Forward pass of the GCN block.
        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices for the graph.
            edge_weight (torch.Tensor, optional): Edge weights for the graph.
            kwargs (dict, optional): Additional keyword arguments for the GCN layer.
        Returns:
            torch.Tensor: Output node features after applying the GCN layer, activation, and skip connection.
        """

        x_res = x
        x = self.conv(
            x, edge_index, edge_weight=edge_weight, **(kwargs if kwargs else {})
        )  # Apply the GCN layer
        x = self.batch_norm(
            x,
        )  # this is a no-op if batch normalization is not used
        x = self.activation(x)
        x = x + self.projection(x_res)  # skip connection
        x = F.dropout(
            x, p=self.dropout, training=self.training
        )  # this is only applied during training
        return x


class GCNBackbone(torch.nn.Module):
    """
    A wrapper class for graph neural networks using a linear chain of multiple GCN blocks.
    """

    def __init__(self, gcn_net: list[GCNBlock]):
        """
        Initializes the GCNBackbone with a list of GCN blocks.
        Args:
            gcn_net (list[GCNBlock]): A list of GCNBlock instances.
        """
        super(GCNBackbone, self).__init__()
        self.gcn_net = torch.nn.ModuleList(gcn_net)

    def forward(self, x, edge_index, edge_weight=None, kwargs=None) -> torch.Tensor:
        """
        Forward pass through the GCN backbone.
        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices for the graph.
            edge_weight (torch.Tensor, optional): Edge weights for the graph.
            kwargs (dict, optional): Additional keyword arguments for the GCN layers.
        Returns:
            torch.Tensor: Output node features after passing through all GCN blocks.
        """
        for layer in self.gcn_net:
            x = layer(x, edge_index, edge_weight=edge_weight, kwargs=kwargs)
        return x


class GCNModel(torch.nn.Module):
    """
    Model for binding together a gcn backbone, a pooling layer, classifier and optionally a graph-feature processor for graph classification tasks.
    """

    def __init__(
        self,
        gcn_net: GCNBackbone,
        classifier: torch.nn.Module,
        pooling_layer: torch.nn.Module,
        graph_features_net: torch.nn.Module | None = None,
    ):
        """
        Initializes the GCNModel.

        Args:
            gcn_net (GCNBackbone): The GCN backbone network.
            classifier (torch.nn.Module): The classifier module.
            pooling_layer (torch.nn.Module): The pooling layer module.
            graph_features_net (torch.nn.Module, optional): The graph features processing module. Defaults to None.
        """
        super(GCNModel, self).__init__()
        self.gcn_net = gcn_net
        self.classifier = classifier
        self.graph_features_net = graph_features_net
        self.pooling_layer = pooling_layer

    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_weight=None,
        graph_features=None,
        gcn_kwargs=None,
    ):
        x = self.gcn_net(
            x, edge_index, edge_weight=edge_weight, **(gcn_kwargs if gcn_kwargs else {})
        )

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        x = self.pooling_layer(x, batch)

        if self.graph_features_net is not None:
            graph_features = self.graph_features_net(graph_features)
            x = torch.cat((x, graph_features), dim=-1)  # last dim

        manifold, boundary, dim = self.classifier(x)

        return manifold, boundary, dim
