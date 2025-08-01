from typing import Any
from collections.abc import Collection

import torch

from . import utils
from . import classifier_block as QGC
from . import graphfeatures_block as QGF
from . import gnn_block as QGGNN


class GNNModel(torch.nn.Module):
    """Torch module for the full GCN model, which consists of a GCN backbone, a classifier, and a pooling layer, augmented with optional graph features network.
    Args:
        torch.nn.Module: base class
    """

    def __init__(
        self,
        gcn_net: list[QGGNN.GNNBlock],
        classifier: QGC.ClassifierBlock,
        pooling_layer: torch.nn.Module,
        graph_features_net: torch.nn.Module = torch.nn.Identity(),
    ):
        """Initialize the GNNModel.

        Args:
            gcn_net (GCNBackbone): GCN backbone network.
            classifier (ClassifierBlock): Classifier block.
            pooling_layer (torch.nn.Module): Pooling layer.
            graph_features_net (torch.nn.Module, optional): Graph features network. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.gcn_net = torch.nn.ModuleList(gcn_net)
        self.classifier = classifier
        self.graph_features_net = graph_features_net
        self.pooling_layer = pooling_layer

    def _eval_gcn_net(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gcn_kwargs: dict[Any, Any] = None,
    ) -> torch.Tensor:
        """Evaluate the GCN network on the input data.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            gcn_kwargs (dict[Any, Any], optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            torch.Tensor: Output of the GCN network.
        """
        features = x
        for gnn_layer in self.gcn_net:
            features = gnn_layer(
                features, edge_index, **(gcn_kwargs if gcn_kwargs else {})
            )
        return features

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        gcn_kwargs: dict = None,
    ):
        """Get embeddings from the GCN model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector for pooling.
            gcn_kwargs (dict, optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            torch.Tensor: Embedding vector for the graph features.
        """
        # apply the GCN backbone to the node features
        embeddings = self._eval_gcn_net(x, edge_index, gcn_kwargs=gcn_kwargs)

        # pool everything together into a single graph representation
        embeddings = self.pooling_layer(embeddings, batch)
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor = None,
        gcn_kwargs: dict[Any, Any] = None,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """Forward run of the gnn model with optional graph features.
        First execute the graph-neural network backbone, then process the graph features, and finally apply the classifier.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector for pooling.
            graph_features (torch.Tensor, optional): Additional graph features. Defaults to None.
            gcn_kwargs (dict[Any, Any], optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            torch.Tensor | Collection[torch.Tensor]: Class predictions.
        """

        # apply the GCN backbone to the node features
        embeddings = self.get_embeddings(x, edge_index, batch, gcn_kwargs=gcn_kwargs)

        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None:
            graph_features = self.graph_features_net(graph_features)
            embeddings = torch.cat(
                (embeddings, graph_features), dim=-1
            )  # -1 -> last dim. This concatenates, but we also could sum them

        # Classifier creates raw the logits
        # no softmax or sigmoid is applied here, as we want to keep the logits for loss calculation
        class_predictions = self.classifier(embeddings)

        return class_predictions

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """
        gcn_net = [QGGNN.GNNBlock.from_config(cfg) for cfg in config["gcn_net"]]
        classifier = QGC.ClassifierBlock.from_config(config["classifier"])
        pooling_layer = utils.get_registered_pooling_layer(config["pooling_layer"])
        graph_features_net = (
            QGF.GraphFeaturesBlock.from_config(config["graph_features_net"])
            if "graph_features_net" in config
            else None
        )

        return cls(
            gcn_net=gcn_net,
            classifier=classifier,
            pooling_layer=pooling_layer,
            graph_features_net=graph_features_net,
        )
