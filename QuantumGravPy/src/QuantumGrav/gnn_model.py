from typing import Any
from collections.abc import Collection

import torch

from . import utils
from . import classifier as QGC
from . import gfeaturesblock as QGF
from . import gnnblock as QGGNN


class GCNModel(torch.nn.Module):
    """Torch module for the full GCN model, which consists of a GCN backbone, a classifier, and a pooling layer, augmented with optional graph features network.
    Args:
        torch.nn.Module: base class
    """

    def __init__(
        self,
        gcn_net: QGC.GNNBlock,
        classifier: QGC.ClassifierBlock,
        pooling_layer: torch.nn.Module,
        graph_features_net: torch.nn.Module = torch.nn.Identity,
    ):
        """Initialize the GCNModel.

        Args:
            gcn_net (GCNBackbone): GCN backbone network.
            classifier (ClassifierBlock): Classifier block.
            pooling_layer (torch.nn.Module): Pooling layer.
            graph_features_net (torch.nn.Module, optional): Graph features network. Defaults to torch.nn.Identity.
        """
        super().__init__()
        self.gcn_net = gcn_net
        self.classifier = classifier
        self.graph_features_net = graph_features_net
        self.pooling_layer = pooling_layer

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        gcn_kwargs: dict = None,
    ):
        """Get embeddings from the GCN model.

        Args:
            x (torch.Tensor): _description_
            edge_index (torch.Tensor): _description_
            batch (torch.Tensor): _description_
            gcn_kwargs (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # apply the GCN backbone to the node features
        x = self.gcn_net(x, edge_index, **(gcn_kwargs if gcn_kwargs else {}))

        # pool everything together into a single graph representation
        x = self.pooling_layer(x, batch)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor = None,
        gcn_kwargs: dict[Any, Any] = None,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            edge_index (torch.Tensor): _description_
            batch (torch.Tensor): _description_
            graph_features (torch.Tensor, optional): _description_. Defaults to None.
            gcn_kwargs (dict[Any, Any], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor | Collection[torch.Tensor]: _description_
        """

        # apply the GCN backbone to the node features
        x = self.get_embeddings(x, edge_index, batch, gcn_kwargs=gcn_kwargs)

        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None:
            graph_features = self.graph_features_net(graph_features)

            x = torch.cat(
                (x, graph_features), dim=-1
            )  # -1 -> last dim. This concatenates, but we also could sum them

        # Classifier creates raw the logits
        # no softmax or sigmoid is applied here, as we want to keep the logits for loss calculation
        class_predictions = self.classifier(x)

        return class_predictions

    @classmethod
    def from_config(cls, config: dict) -> "GCNModel":
        """Create a GCNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GCNModel: An instance of GCNModel.
        """
        gcn_net = QGGNN.GNNBlock.from_config(config["gcn_net"])
        classifier = QGC.ClassifierBlock.from_config(config["classifier"])
        pooling_layer = utils.get_pooling_layer(config["pooling_layer"])
        graph_features_net = (
            QGF.GraphFeaturesBlock.from_config(config["graph_features_net"])
            if "graph_features_net" in config
            else torch.nn.Identity()
        )

        return cls(
            gcn_net=gcn_net,
            classifier=classifier,
            pooling_layer=pooling_layer,
            graph_features_net=graph_features_net,
        )
