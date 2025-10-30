from typing import Any, Callable, Sequence
from pathlib import Path
from inspect import isclass
import torch
from . import utils
from . import linear_sequential as QGLS
from . import gnn_block as QGGNN


class ModuleWrapper(torch.nn.Module):
    """Wrapper to make pooling functions compatible with ModuleList."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def get_fn(self) -> Callable:
        return self.fn


class GNNModel(torch.nn.Module):
    """Torch module for the full GCN model, which consists of a GCN backbone, a set of downstream tasks, and a pooling layer, augmented with optional graph features network.
    Args:
        torch.nn.Module: base class
    """

    # FIXME: change this such that all elements are build internally, and only types are passed in
    # this is more in line with the other parts of the system and also with how the configs work
    def __init__(
        self,
        encoder: Sequence[QGGNN.GNNBlock],
        downstream_tasks: Sequence[torch.nn.Module],
        pooling_layers: Sequence[torch.nn.Module] | None = None,
        aggregate_pooling: torch.nn.Module | Callable | None = None,
        graph_features_net: torch.nn.Module | None = None,
        aggregate_graph_features: torch.nn.Module | Callable | None = None,
        active_tasks: list[int] | None = None,
    ):
        """Initialize the GNNModel.

        Args:
            encoder (GCNBackbone): GCN backbone network.
            downstream_tasks (Sequence[torch.nn.Module]): Downstream task blocks. These are assumed to be independent of each other.
            pooling_layers (Sequence[torch.nn.Module]): Pooling layers. Defaults to None.
            aggregate_pooling (torch.nn.Module | Callable | None): Aggregation of pooling layer output. Defaults to None.
            graph_features_net (torch.nn.Module, optional): Graph features network. Defaults to None.
            aggregate_graph_features (torch.nn.Module | Callable | None): Aggregation of graph features. Defaults to None.
        """
        super().__init__()

        # encoder is a sequence of GNN blocks. There must be at least one
        self.encoder = torch.nn.ModuleList(encoder)

        if len(self.encoder) == 0:
            raise ValueError("At least one GNN block must be provided.")

        # set up downstream tasks. These are independent of each other, but there must be one at least
        self.downstream_tasks = torch.nn.ModuleList(downstream_tasks)

        if len(self.downstream_tasks) == 0:
            raise ValueError("At least one downstream task must be provided.")

        if active_tasks is None:
            raise ValueError("active_tasks must be provided.")
        else:
            self.active_tasks = active_tasks

        # set up pooling layers and their aggregation
        if pooling_layers is not None:
            if len(pooling_layers) == 0:
                raise ValueError("At least one pooling layer must be provided.")

            self.pooling_layers = torch.nn.ModuleList(
                [
                    p
                    if isclass(type(p)) and issubclass(type(p), torch.nn.Module)
                    else ModuleWrapper(p)
                    for p in pooling_layers
                ]
            )
        else:
            self.pooling_layers = None

        # aggregate pooling layer
        self.aggregate_pooling = aggregate_pooling

        if aggregate_pooling is not None:
            if not isclass(aggregate_pooling) or not issubclass(
                aggregate_pooling, torch.nn.Module
            ):
                self.aggregate_pooling = ModuleWrapper(aggregate_pooling)

        pooling_funcs = [self.aggregate_pooling, self.pooling_layers]
        if any([p is not None for p in pooling_funcs]) and not all(
            p is not None for p in pooling_funcs
        ):
            raise ValueError(
                "If pooling layers are to be used, both an aggregate pooling method and pooling layers must be provided."
            )

        # set up graph features processing if provided
        self.graph_features_net = graph_features_net

        self.aggregate_graph_features = aggregate_graph_features

        if aggregate_graph_features is not None:
            if not isclass(aggregate_graph_features) or not issubclass(
                aggregate_graph_features, torch.nn.Module
            ):
                self.aggregate_graph_features = ModuleWrapper(aggregate_graph_features)

        graph_processors = [self.graph_features_net, self.aggregate_graph_features]

        if any([g is not None for g in graph_processors]) and not all(
            g is not None for g in graph_processors
        ):
            raise ValueError(
                "If graph features are to be used, both a graph features network and an aggregation method must be provided."
            )

    def set_task_active(self, i: int) -> None:
        """Set a downstream task as active.

        Args:
            i (int): Index of the downstream task to activate.
        """

        if i < 0 or i >= len(self.active_tasks):
            raise ValueError("Invalid task index.")

        self.active_tasks[i] = True

    def set_task_inactive(self, i: int) -> None:
        """Set a downstream task as inactive.

        Args:
            i (int): Index of the downstream task to deactivate.
        """

        if i < 0 or i >= len(self.active_tasks):
            raise ValueError("Invalid task index.")

        self.active_tasks[i] = False

    def eval_encoder(
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
        embeddings = self.eval_encoder(
            x, edge_index, **(gcn_kwargs if gcn_kwargs else {})
        )

        # pool everything together into a single graph representation
        if self.pooling_layers is not None and self.aggregate_pooling is not None:
            pooled_embeddings = [
                pooling_op(embeddings, batch) for pooling_op in self.pooling_layers
            ]

            return self.aggregate_pooling(pooled_embeddings)
        else:
            return embeddings

    def compute_downstream_tasks(
        self,
        x: torch.Tensor,
        downstream_task_args: Sequence[tuple | list] | None = None,
        downstream_task_kwargs: Sequence[dict] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Compute the outputs of the downstream tasks. Only the active tasks will be computed.

        Args:
            x (torch.Tensor): Input embeddings tensor
            downstream_task_args (Sequence[tuple | list] | None, optional): Arguments for downstream tasks. Defaults to None.
            downstream_task_kwargs (Sequence[dict] | None, optional): Keyword arguments for downstream tasks. Defaults to None.

        Returns:
            dict[int, torch.Tensor]: Outputs of the downstream tasks.
        """

        output = {}

        for i in range(len(self.downstream_tasks)):
            if self.active_tasks[i]:
                task = self.downstream_tasks[i]

                task_args = []
                task_kwargs = {}
                if (
                    downstream_task_args is not None
                    and i < len(downstream_task_args)
                    and downstream_task_args[i]
                ):
                    task_args = downstream_task_args[i]

                if (
                    downstream_task_kwargs is not None
                    and i < len(downstream_task_kwargs)
                    and downstream_task_kwargs[i]
                ):
                    task_kwargs = downstream_task_kwargs[i]

                res = task(x, *task_args, **task_kwargs)
                output[i] = res

        return output

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        downstream_task_args: Sequence[tuple | list] | None = None,
        downstream_task_kwargs: Sequence[dict] | None = None,
        embedding_kwargs: dict[Any, Any] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Forward run of the gnn model with optional graph features.
        # First execute the graph-neural network backbone, then process the graph features, and finally apply the downstream tasks.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector for pooling.
            graph_features (torch.Tensor | None, optional): Additional graph features. Defaults to None.
            downstream_task_args (Sequence[tuple] | None, optional): Arguments for downstream tasks. Defaults to None.
            downstream_task_kwargs (Sequence[dict] | None, optional): Keyword arguments for downstream tasks. Defaults to None.
            embedding_kwargs (dict[Any, Any] | None, optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            Sequence[torch.Tensor]: Raw output of downstream tasks.
        """
        # apply the GCN backbone to the node features

        embeddings = self.get_embeddings(
            x, edge_index, batch, gcn_kwargs=embedding_kwargs
        )

        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None and self.graph_features_net is not None:
            graph_features = self.graph_features_net(graph_features)

        if self.aggregate_graph_features is not None and graph_features is not None:
            embeddings = self.aggregate_graph_features(embeddings, graph_features)

        # downstream tasks are given out as is, no softmax or other assumptions
        return self.compute_downstream_tasks(
            embeddings,
            downstream_task_args=downstream_task_args,
            downstream_task_kwargs=downstream_task_kwargs,
        )

    @classmethod
    def _cfg_helper(
        cls, cfg: dict[str, Any], utility_function: Callable, throw_message: str
    ) -> torch.nn.Module | Callable:
        """Helper function to create a module or callable from a config.

        Args:
            cfg (dict[str, Any]): config node. must contain type, args, kwargs
            utility_function (Callable): utility function to create the module or callable
            throw_message (str): message to throw in case of error

        Raises:
            ValueError: if the config node is invalid
            ValueError: if the utility function could not find the specified type

        Returns:
            torch.nn.Module | Callable: created module or callable
        """
        if not utils.verify_config_node(cfg):
            raise ValueError(throw_message)
        f = utility_function(cfg["type"])

        if f is None:
            raise ValueError(
                f"Utility function '{utility_function.__name__}' could not find '{cfg['type']}'"
            )

        if isinstance(f, type):
            return f(
                cfg["args"] if "args" in cfg else [],
                **(cfg["kwargs"] if "kwargs" in cfg else {}),
            )
        else:
            return f

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """

        # create encoder
        encoder = [QGGNN.GNNBlock.from_config(cfg) for cfg in config["encoder"]]

        # create downstream tasks
        downstream_tasks = [
            QGLS.LinearSequential.from_config(cfg) for cfg in config["downstream_tasks"]
        ]  # TODO: generalize this!

        # make pooling layers
        pooling_layers_cfg = config.get("pooling_layers", None)

        if pooling_layers_cfg is not None:
            pooling_layers = []
            for pool_cfg in pooling_layers_cfg:
                pooling_layer = cls._cfg_helper(
                    pool_cfg,
                    utils.get_registered_pooling_layer,
                    f"The config for a pooling layer is invalid: {pool_cfg}",
                )
                pooling_layers.append(pooling_layer)
        else:
            pooling_layers = None

        # graph aggregation pooling
        aggregate_pooling_cfg = config.get("aggregate_pooling", None)
        if aggregate_pooling_cfg is not None:
            aggregate_pooling = cls._cfg_helper(
                config["aggregate_pooling"],
                utils.get_pooling_aggregation,
                f"The config for 'aggregate_pooling' is invalid: {config['aggregate_pooling']}",
            )
        else:
            aggregate_pooling = None

        # make graph features network and aggregations
        if "graph_features_net" in config and config["graph_features_net"] is not None:
            graph_features_net = QGLS.LinearSequential.from_config(
                config["graph_features_net"]
            )
        else:
            graph_features_net = None

        if graph_features_net is not None:
            aggregate_graph_features = cls._cfg_helper(
                config["aggregate_graph_features"],
                utils.get_graph_features_aggregation,
                f"The config for 'aggregate_graph_features' is invalid: {config['aggregate_graph_features']}",
            )
        else:
            aggregate_graph_features = None

        active_tasks = [cfg.get("active", False) for cfg in config["downstream_tasks"]]

        # return the model
        return cls(
            encoder=encoder,
            downstream_tasks=downstream_tasks,
            pooling_layers=pooling_layers,
            graph_features_net=graph_features_net,
            aggregate_graph_features=aggregate_graph_features,
            aggregate_pooling=aggregate_pooling,
            active_tasks=active_tasks,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize the model to a config

        Returns:
            dict[str, Any]: _description_
        """
        pooling_layer_names = None
        if self.pooling_layers is not None:
            pooling_layer_names = []
            for layer in self.pooling_layers:
                if isinstance(layer, ModuleWrapper):
                    pooling_layer_names.append(
                        {
                            "type": utils.pooling_layers_names[layer.get_fn()],
                            "args": [],
                            "kwargs": {},
                        }
                    )
                else:
                    pooling_layer_names.append(
                        {
                            "type": utils.pooling_layers_names[layer],
                            "args": [],
                            "kwargs": {},
                        }
                    )

        aggregate_graph_features_names = None
        if self.aggregate_graph_features is not None:
            if isinstance(self.aggregate_graph_features, ModuleWrapper):
                aggregate_graph_features_names = {
                    "type": utils.graph_features_aggregations_names[
                        self.aggregate_graph_features.get_fn()
                    ],
                    "args": [],
                    "kwargs": {},
                }
            else:
                aggregate_graph_features_names = {
                    "type": utils.graph_features_aggregations_names[
                        self.aggregate_graph_features
                    ],
                    "args": [],
                    "kwargs": {},
                }

        aggregate_pooling_names = None
        if self.aggregate_pooling is not None:
            if isinstance(self.aggregate_pooling, ModuleWrapper):
                aggregate_pooling_names = {
                    "type": utils.pooling_aggregations_names[
                        self.aggregate_pooling.get_fn()
                    ],
                    "args": [],
                    "kwargs": {},
                }
            else:
                aggregate_pooling_names = {
                    "type": utils.pooling_aggregations_names[self.aggregate_pooling],
                    "args": [],
                    "kwargs": {},
                }

        # downstream_task_configs

        downstream_task_configs = [task.to_config() for task in self.downstream_tasks]
        for i in range(len(self.downstream_tasks)):
            downstream_task_configs[i]["active"] = self.active_tasks[i]

        config = {
            "encoder": [encoder_layer.to_config() for encoder_layer in self.encoder],
            "downstream_tasks": downstream_task_configs,
            "pooling_layers": pooling_layer_names,
            "graph_features_net": self.graph_features_net.to_config()
            if self.graph_features_net
            else None,
            "aggregate_graph_features": aggregate_graph_features_names,
            "aggregate_pooling": aggregate_pooling_names,
            "active_tasks": self.active_tasks,
        }

        return config

    def save(self, path: str | Path) -> None:
        """Save the model state to file. This saves a dictionary structured like this:
         'encoder': self.encoder,
         'downstream_tasks': self.downstream_tasks,
         'pooling_layers': self.pooling_layers,
         'graph_features_net': self.graph_features_net,
         'aggregate_graph_features': self.aggregate_graph_features,
         'aggregate_pooling': self.aggregate_pooling,

        Args:
            path (str | Path): Path to save the model to
        """

        config = self.to_config()

        torch.save(
            {"config": config, "model": self.state_dict()},
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
        model_dict = torch.load(path, weights_only=False)
        model = cls.from_config(model_dict["config"]).to(device)
        model.load_state_dict(model_dict["model"])

        return model
