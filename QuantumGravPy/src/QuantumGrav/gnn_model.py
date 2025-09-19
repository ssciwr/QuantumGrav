from typing import Any, Callable
from collections.abc import Collection
from pathlib import Path

import torch

from . import utils
from . import classifier_block as QGC
from . import graphfeatures_block as QGF
from . import gnn_block as QGGNN


class GNNModel(torch.nn.Module):
    """Torch module for the full GCN model, which consists of a GCN backbone, a set of downstream tasks, and a pooling layer, augmented with optional graph features network.
    Args:
        torch.nn.Module: base class
    """

    def __init__(
        self,
        encoder: list[QGGNN.GNNBlock],
        downstream_tasks: list[torch.nn.Module],
        pooling_layers: list[torch.nn.Module],
        aggregate_pooling: torch.nn.Module | Callable | None = None,
        graph_features_net: torch.nn.Module | None = None,
        aggregate_graph_features: torch.nn.Module | Callable | None = None,
    ):
        """Initialize the GNNModel.

        Args:
            encoder (GCNBackbone): GCN backbone network.
            downstream_tasks (list[torch.nn.Module]): Downstream task blocks.
            pooling_layers (list[torch.nn.Module]): Pooling layers. Defaults to None.
            aggregate_pooling (torch.nn.Module | Callable | None): Aggregation of pooling layer output. Defaults to None.
            graph_features_net (torch.nn.Module, optional): Graph features network. Defaults to None.
            aggregate_graph_features (torch.nn.Module | Callable | None): Aggregation of graph features. Defaults to None.
        """
        super().__init__()
        self.encoder = torch.nn.ModuleList(encoder)
        self.downstream_tasks = downstream_tasks
        self.graph_features_net = graph_features_net
        self.pooling_layers = pooling_layers
        self.aggregate_pooling = aggregate_pooling
        self.aggregate_graph_features = aggregate_graph_features

        if self.aggregate_pooling is None and len(self.pooling_layers) > 1:
            raise ValueError(
                "If multiple pooling layers are defined, an aggregation method must be provided."
            )

    def _eval_encoder(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gcn_kwargs: dict[Any, Any] | None = None,
    ) -> torch.Tensor:
        """Evaluate the GCN network on the input data.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            gcn_kwargs (dict[Any, Any], optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            torch.Tensor: Output of the GCN network.
        """
        # Apply each GCN layer to the input features
        features = x
        for gnn_layer in self.encoder:
            features = gnn_layer(
                features, edge_index, **(gcn_kwargs if gcn_kwargs else {})
            )
        return features

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_kwargs: dict | None = None,
    ) -> torch.Tensor:
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
        embeddings = self._eval_encoder(
            x, edge_index, **(gcn_kwargs if gcn_kwargs else {})
        )

        # pool everything together into a single graph representation
        pooled_embeddings = [
            pooling_op(embeddings, batch) for pooling_op in self.pooling_layers
        ]
        if self.aggregate_pooling is not None:
            return self.aggregate_pooling(pooled_embeddings)
        else:
            return pooled_embeddings[0]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        downstream_task_args: list[tuple | list] | None = None,
        downstream_task_kwargs: list[dict] | None = None,
        gcn_kwargs: dict[Any, Any] | None = None,
    ) -> Collection[torch.Tensor]:
        """Forward run of the gnn model with optional graph features.
        # First execute the graph-neural network backbone, then process the graph features, and finally apply the downstream tasks.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector for pooling.
            graph_features (torch.Tensor | None, optional): Additional graph features. Defaults to None.
            downstream_task_args (list[tuple] | None, optional): Arguments for downstream tasks. Defaults to None.
            downstream_task_kwargs (list[dict] | None, optional): Keyword arguments for downstream tasks. Defaults to None.
            gcn_kwargs (dict[Any, Any] | None, optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            Collection[torch.Tensor]: Raw output of downstream tasks.
        """
        # apply the GCN backbone to the node features
        embeddings = self.get_embeddings(x, edge_index, batch, gcn_kwargs=gcn_kwargs)

        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None:
            if (
                self.graph_features_net is not None
                and self.aggregate_graph_features is not None
            ):
                graph_features = self.graph_features_net(graph_features)
                embeddings = self.aggregate_graph_features(embeddings, graph_features)
            else:
                raise ValueError(
                    "Graph features network or aggregation function is not defined, but graph features should be used"
                )

        # downstream tasks are given out as is, no softmax or other assumptions
        output = []

        if downstream_task_args is not None:
            downstream_task_args = [
                downstream_task_args[i] if downstream_task_args[i] is not None else []
                for i in range(1, len(self.downstream_tasks))
            ]
        else:
            downstream_task_args = [[] for _ in self.downstream_tasks]

        if downstream_task_kwargs is not None:
            downstream_task_kwargs = [
                downstream_task_kwargs[i]
                if downstream_task_kwargs[i] is not None
                else {}
                for i in range(1, len(self.downstream_tasks))
            ]
        else:
            downstream_task_kwargs = [{} for _ in self.downstream_tasks]

        for downstream_task, downstream_task_args, downstream_task_kwargs in zip(
            self.downstream_tasks,
            downstream_task_args,
            downstream_task_kwargs,
        ):
            res = downstream_task(
                embeddings, *downstream_task_args, **downstream_task_kwargs
            )
            output.append(res)

        return output

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """
        encoder = [QGGNN.GNNBlock.from_config(cfg) for cfg in config["encoder"]]
        downstream_tasks = [
            QGC.ClassifierBlock.from_config(cfg) for cfg in config["downstream_tasks"]
        ]  # TODO: generalize this!
        pooling_layers = [
            utils.get_registered_pooling_layer(cfg) for cfg in config["pooling_layers"]
        ]
        graph_features_net = (
            QGF.GraphFeaturesBlock.from_config(config["graph_features_net"])
            if "graph_features_net" in config
            and config["graph_features_net"] is not None
            else None
        )

        return cls(
            encoder=encoder,
            downstream_tasks=downstream_tasks,
            pooling_layers=pooling_layers,
            graph_features_net=graph_features_net,
        )

    def save(self, path: str | Path) -> None:
        """Save the model state to file. This saves a dictionary structured like this:
         'encoder': self.encoder,
         'classifier': self.classifier,
         'pooling_layer': self.pooling_layer,
         'graph_features_net': self.graph_features_net

        Args:
            path (str | Path): Path to save the model to
        """
        torch.save(
            {
                "encoder": self.encoder,
                "downstream_tasks": self.downstream_tasks,
                "pooling_layers": self.pooling_layers,
                "graph_features_net": self.graph_features_net,
                "aggregate_graph_features": self.aggregate_graph_features,
                "aggregate_pooling": self.aggregate_pooling,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device = torch.device("cpu")
    ) -> "GNNModel":
        """Load a model from file that has previously been save with the function 'save'.

        Args:
            path (str | Path): path to load the model from.
            device (torch.device): device to put the model to. Defaults to torch.device("cpu")
        Returns:
            GNNModel: model instance initialized with the sub-models loaded from file.
        """
        model_dict = torch.load(path, map_location=device, weights_only=False)

        return cls(
            model_dict["encoder"],
            model_dict["downstream_tasks"],
            model_dict["pooling_layers"],
            model_dict["aggregate_pooling"],
            model_dict["graph_features_net"],
            model_dict["aggregate_graph_features"],
        )
