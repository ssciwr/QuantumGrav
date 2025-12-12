from typing import Any, Callable, Sequence, Dict, Tuple
from pathlib import Path
from inspect import isclass
import jsonschema

import torch
from collections.abc import Collection

from . import base


class ModuleWrapper(torch.nn.Module):
    """Wrapper to make pooling functions compatible with ModuleList and ModuleDict"""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def get_fn(self) -> Callable:
        return self.fn


def instantiate_type(
    object_or_type: type | torch.nn.Module | Callable,
    args: Sequence[Any] | None,
    kwargs: Dict[str, Any] | None,
):
    """Helper to instantiate a type from args, kwargs or use it directly. When a function is passed, it will be wrapped in a `ModuleWrapper` instance

    Args:
        object_or_type (type | torch.nn.Module): type or object to check and instantiate
        args (Sequence[Any] | None): args to build the object
        kwargs (Dict[str, Any] | None): kwargs to build the object

    Raises:
        ValueError: When the type is not a subclass or instance of both torch.nn.Module and QG.base.Configurable

    Returns:
         newly instantiated object of type 'object_or_type' or the passed object
    """

    if isinstance(object_or_type, torch.nn.Module):
        return object_or_type
    elif isclass(object_or_type) and issubclass(object_or_type, torch.nn.Module):
        return object_or_type(*(args if args else []), **(kwargs if kwargs else {}))
    elif callable(object_or_type):
        return ModuleWrapper(object_or_type)
    else:
        raise ValueError(
            f"{object_or_type} must be either a subtype of torch.nn.Module or an instance of such a type or a callable"
        )


class GNNModel(torch.nn.Module, base.Configurable):
    """Complete GNN model architecture with encoder, pooling, and downstream tasks.

    This model combines:
    - An encoder network (typically GNN layers) to process graph structure
    - Optional pooling layers to aggregate node features into graph-level representations
    - Multiple downstream task heads for classification, regression, etc.
    - Optional graph features network for processing additional graph-level features

    The model supports multi-task learning with selective task activation.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GNNModel Configuration",
        "type": "object",
        "properties": {
            "encoder_type": {
                "description": "Type of the encoder network (Python class or string)",
            },
            "encoder_args": {
                "type": "array",
                "description": "Positional arguments for encoder initialization",
                "items": {},
            },
            "encoder_kwargs": {
                "type": "object",
                "description": "Keyword arguments for encoder initialization",
            },
            "downstream_tasks": {
                "type": "array",
                "description": "List of downstream tasks, each as [type, args, kwargs]",
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": [
                        {"description": "Task type (class or string)"},
                        {"type": "array", "description": "Task args", "items": {}},
                        {"type": "object", "description": "Task kwargs"},
                    ],
                },
                "minItems": 1,
            },
            "pooling_layers": {
                "type": "array",
                "description": "List of pooling layers, each as [type, args, kwargs]",
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": [
                        {"description": "Pooling type (class, callable, or string)"},
                        {"type": "array", "description": "Pooling args", "items": {}},
                        {"type": "object", "description": "Pooling kwargs"},
                    ],
                },
            },
            "aggregate_pooling_type": {
                "description": "Type for aggregating pooling results (class, callable, or string)",
            },
            "aggregate_pooling_args": {
                "type": "array",
                "description": "Arguments for aggregate pooling",
                "items": {},
            },
            "aggregate_pooling_kwargs": {
                "type": "object",
                "description": "Keyword arguments for aggregate pooling",
            },
            "latent_model_type": {
                "description": "Type of a general latent space model. Alternative to pooling_layers + aggregate_pooling"
            },
            "latent_model_args": {
                "type": "array",
                "description": "Arguments for latent model",
                "items": {},
            },
            "latent_model_kwargs": {
                "type": "object",
                "description": "Keyword arguments for latent model",
            },
            "graph_features_net_type": {
                "description": "Type of graph features network (class or string)",
            },
            "graph_features_net_args": {
                "type": "array",
                "description": "Arguments for graph features network",
                "items": {},
            },
            "graph_features_net_kwargs": {
                "type": "object",
                "description": "Keyword arguments for graph features network",
            },
            "aggregate_graph_features_type": {
                "description": "Type for aggregating graph features (class, callable, or string)",
            },
            "aggregate_graph_features_args": {
                "type": "array",
                "description": "Arguments for aggregate graph features",
                "items": {},
            },
            "aggregate_graph_features_kwargs": {
                "type": "object",
                "description": "Keyword arguments for aggregate graph features",
            },
            "active_tasks": {
                "type": "object",
                "description": "Dictionary mapping task indices/names to boolean active status",
            },
        },
        "required": [
            "encoder_type",
            "downstream_tasks",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        encoder_type: type | torch.nn.Module,
        downstream_tasks: Sequence[
            Tuple[type | torch.nn.Module, Sequence[Any] | None, Dict[str, Any] | None]
        ],
        encoder_args: Sequence[Any] | None = None,
        encoder_kwargs: Dict[str, Any] | None = None,
        pooling_layers: Sequence[
            Tuple[type | torch.nn.Module, Sequence[Any] | None, Dict[str, Any] | None]
        ]
        | None = None,
        aggregate_pooling_type: type | torch.nn.Module | Callable | None = None,
        aggregate_pooling_args: Sequence[Any] | None = None,
        aggregate_pooling_kwargs: Dict[str, Any] | None = None,
        latent_model_type: type | torch.nn.Module | None = None,
        latent_model_args: Sequence[Any] | None = None,
        latent_model_kwargs: Dict[str, Any] | None = None,
        graph_features_net_type: type | torch.nn.Module | None = None,
        graph_features_net_args: Sequence[Any] | None = None,
        graph_features_net_kwargs: Dict[str, Any] | None = None,
        aggregate_graph_features_type: type | torch.nn.Module | Callable | None = None,
        aggregate_graph_features_args: Sequence[Any] | None = None,
        aggregate_graph_features_kwargs: Dict[str, Any] | None = None,
        active_tasks: Dict[int, bool] | None = None,
    ):
        """Initialize GNNModel with encoder, pooling, and downstream task components.

        Args:
            encoder_type (type): Class type or torch Module instance for the encoder network (e.g., GNN backbone).
            encoder_args (Sequence[Any]): Positional arguments to pass to encoder_type constructor.
            encoder_kwargs (Dict[str, Any]): Keyword arguments to pass to encoder_type constructor.
            downstream_tasks (Sequence[Sequence[type, Sequence[Any], Dict[str, Any]]]): List of downstream tasks,
                where each task is specified as [task_type, task_args, task_kwargs].
            pooling_layers (Sequence[Sequence[type, Sequence[Any], Dict[str, Any]]] | None, optional): List of pooling layers,
                where each layer is specified as [pooling_type, pooling_args, pooling_kwargs]. Defaults to None.
            aggregate_pooling_type (type | Callable | None, optional): Type, Module instance or function for aggregating multiple pooling outputs.
                Required if pooling_layers is provided. Defaults to None.
            aggregate_pooling_args (Sequence[Any] | None, optional): Positional arguments for aggregate_pooling_type. Defaults to None.
            aggregate_pooling_kwargs (Dict[str, Any] | None, optional): Keyword arguments for aggregate_pooling_type. Defaults to None.
            latent_model_type (type | torch.nn.Module | None): Latent model type. Either this or pooling_layers can be used, not both.
            latent_model_args (Sequence[Any] | None, optional): Latent model args.
            latent_model_kwargs (Dict[str, Any] | None, optional): Latent model kwargs.
            graph_features_net_type (type | None, optional): Network type for processing additional graph-level features. Defaults to None.
            graph_features_net_args (Sequence[Any] | None, optional): Positional arguments for graph_features_net_type. Defaults to None.
            graph_features_net_kwargs (Dict[str, Any] | None, optional): Keyword arguments for graph_features_net_type. Defaults to None.
            aggregate_graph_features_type (type | Callable | None, optional): Type, Module instance or function for combining embeddings with graph features.
                Required if graph_features_net_type is provided. Defaults to None.
            aggregate_graph_features_args (Sequence[Any] | None, optional): Positional arguments for aggregate_graph_features_type. Defaults to None.
            aggregate_graph_features_kwargs (Dict[str, Any] | None, optional): Keyword arguments for aggregate_graph_features_type. Defaults to None.
            active_tasks (Dict[int, bool] | None, optional): Dictionary mapping task indices to active status.
                If None, all tasks are active by default. Defaults to None.

        Raises:
            ValueError: If downstream_tasks is empty (at least one task required).
            ValueError: If pooling_layers provided without aggregate_pooling_type or vice versa.
            ValueError: If pooling_layers is empty when provided.
            ValueError: If graph_features_net_type provided without aggregate_graph_features_type or vice versa.
            ValueError: If pooling_layers and latent_type are given at the same time.
        """

        # check consistency
        graph_processors = [graph_features_net_type, aggregate_graph_features_type]
        if any([g is not None for g in graph_processors]) and not all(
            g is not None for g in graph_processors
        ):
            raise ValueError(
                "If graph features are to be used, both a graph features network and an aggregation method must be provided."
            )

        pooling_funcs = [aggregate_pooling_type, pooling_layers]
        self.with_pooling = False
        if any(p is not None for p in pooling_funcs):
            if not all(p is not None for p in pooling_funcs):
                raise ValueError(
                    "If pooling layers are to be used, both an aggregate pooling method and pooling layers must be provided."
                )
            else:
                self.with_pooling = True

        self.with_latent = False
        if latent_model_type is not None:
            self.with_latent = True

        if self.with_latent and self.with_pooling:
            raise ValueError(
                "Either latent_model_type or pooling_layers and aggregate_pooling can be given, not both"
            )

        # set up downstream tasks. These are independent of each other, but there must be one at least
        if len(downstream_tasks) == 0:
            raise ValueError("At least one downstream task must be provided.")

        super().__init__()

        self.encoder_args = encoder_args
        self.encoder_kwargs = encoder_kwargs
        self.downstream_task_specs = downstream_tasks
        self.pooling_layer_specs = pooling_layers

        self.aggregate_pooling_type = aggregate_pooling_type
        self.aggregate_pooling_args = aggregate_pooling_args
        self.aggregate_pooling_kwargs = aggregate_pooling_kwargs

        self.graph_features_net_args = graph_features_net_args
        self.graph_features_net_kwargs = graph_features_net_kwargs

        self.aggregate_graph_features_type = aggregate_graph_features_type
        self.aggregate_graph_features_args = aggregate_graph_features_args
        self.aggregate_graph_features_kwargs = aggregate_graph_features_kwargs

        # set up encoder type and downstream tasks
        self.encoder = instantiate_type(encoder_type, encoder_args, encoder_kwargs)

        downstream_task_modules = []
        for task_type, args, kwargs in downstream_tasks:
            downstream_task_modules.append(instantiate_type(task_type, args, kwargs))

        self.downstream_tasks = torch.nn.ModuleList(downstream_task_modules)

        if self.with_pooling:
            if pooling_layers is not None:
                if len(pooling_layers) == 0:
                    raise ValueError("At least one pooling layer must be provided.")

                self.pooling_layers = torch.nn.ModuleList(
                    [
                        instantiate_type(pl_type, args, kwargs)
                        for pl_type, args, kwargs in pooling_layers
                    ]
                )
            else:
                self.pooling_layers = None

            # aggregate pooling layer
            if aggregate_pooling_type is not None:
                self.aggregate_pooling = instantiate_type(
                    aggregate_pooling_type,
                    aggregate_pooling_args,
                    aggregate_pooling_kwargs,
                )
            else:
                self.aggregate_pooling = torch.nn.Identity()  # non-op

        if self.with_latent:
            # alternatively to pooling_layer, set up latent_model
            self.latent_model = instantiate_type(
                latent_model_type, latent_model_args, latent_model_kwargs
            )

        # set up graph features processing if provided
        if graph_features_net_type is not None:
            self.graph_features_net = instantiate_type(
                graph_features_net_type,
                graph_features_net_args,
                graph_features_net_kwargs,
            )
        else:
            self.graph_features_net = torch.nn.Identity()  # non-op

        if aggregate_graph_features_type is not None:
            self.aggregate_graph_features = instantiate_type(
                aggregate_graph_features_type,
                aggregate_graph_features_args,
                aggregate_graph_features_kwargs,
            )
        else:
            self.aggregate_graph_features = torch.nn.Identity()  # non-op

        # active tasks
        if active_tasks:
            if len(active_tasks) != len(self.downstream_tasks) or set(
                active_tasks.keys()
            ) != set(range(len(self.downstream_tasks))):
                raise ValueError(
                    "active_tasks keys must match the indices of downstream tasks."
                )
            self.active_tasks: Dict[int, bool] = active_tasks
        else:
            self.active_tasks = {i: True for i in range(0, len(self.downstream_tasks))}

    def set_task_active(self, key: int) -> None:
        """Set a downstream task as active.

        Args:
            key (int): key (name) of the downstream task to activate.
        """
        if key not in self.active_tasks:
            raise KeyError(f"Task {key} not found in active tasks.")
        self.active_tasks[key] = True

    def set_task_inactive(self, key: int) -> None:
        """Set a downstream task as inactive.

        Args:
            key (int): key (name) of the downstream task to deactivate.
        """

        if key not in self.active_tasks:
            raise KeyError(f"Task {key} not found in active tasks.")
        self.active_tasks[key] = False

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_kwargs: Dict[str, Any] | None = None,
        latent_args: Sequence[Any] | None = None,
        latent_kwargs: Dict[str, Any] | None = None,
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
        embeddings = self.encoder(x, edge_index, **(gcn_kwargs if gcn_kwargs else {}))

        # pool everything together into a single graph representation
        if self.with_pooling:
            if not self.pooling_layers or self.pooling_layers == [None]:
                # No pooling layers provided; pass embeddings directly
                pooled_embeddings = [
                    embeddings,
                ]
            else:
                pooled_embeddings = [
                    pooling_op(embeddings, batch) if pooling_op else embeddings
                    for pooling_op in self.pooling_layers
                ]

            return self.aggregate_pooling(pooled_embeddings)

        elif self.with_latent:
            return self.latent_model(
                embeddings,
                *(latent_args if latent_args is not None else []),
                **(latent_kwargs if latent_kwargs is not None else {}),
            )

    def compute_downstream_tasks(
        self,
        x: torch.Tensor,
        args: Sequence[Tuple | Sequence] | None = None,
        kwargs: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[int, torch.Tensor | Collection[torch.Tensor]]:
        """Compute the outputs of the downstream tasks. Only the active tasks will be computed.

        Args:
            x (torch.Tensor): Input embeddings tensor
            args (Sequence[Tuple | Sequence] | None, optional): Arguments for downstream tasks. Defaults to None.
            kwargs (Sequence[Dict[str, Any]] | None, optional): Keyword arguments for downstream tasks. Defaults to None.

        Returns:
            Dict[int, torch.Tensor | Collection[torch.Tensor]]: Outputs of the downstream tasks.
        """
        d_args = [[] for _ in self.downstream_tasks] if args is None else args
        d_kwargs: Sequence[Dict[str, Any]] = (
            [{} for _ in self.downstream_tasks] if kwargs is None else kwargs
        )
        return {
            i: self.downstream_tasks[i](x, *d_args[i], **d_kwargs[i])
            for i in range(0, len(self.downstream_tasks))
            if self.active_tasks[i]
        }

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_features: torch.Tensor | None = None,
        latent_args: Sequence[Any] | None = None,
        latent_kwargs: Dict[str, Any] | None = None,
        downstream_task_args: Sequence[Tuple | Sequence[Any]] | None = None,
        downstream_task_kwargs: Sequence[Dict[str, Any]] | None = None,
        embedding_kwargs: Dict[Any, Any] | None = None,
    ) -> Dict[int, torch.Tensor]:
        """Forward run of the gnn model with optional graph features.
        First execute the graph-neural network backbone, then process the graph features, and finally apply the downstream tasks.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector for pooling.
            graph_features (torch.Tensor | None, optional): Additional graph features. Defaults to None.
            downstream_task_args ( Sequence[Tuple | Sequence[Any]] | None, optional): Arguments for downstream tasks. Defaults to None.
            downstream_task_kwargs (Sequence[Dict[str, Any]] | None, optional): Keyword arguments for downstream tasks. Defaults to None.
            embedding_kwargs (dict[Any, Any] | None, optional): Additional arguments for the GCN. Defaults to None.

        Returns:
            Dict[int, torch.Tensor]: Raw output of downstream tasks.
        """
        # apply the GCN backbone to the node features
        embeddings = self.get_embeddings(
            x,
            edge_index,
            batch,
            gcn_kwargs=embedding_kwargs,
            latent_args=latent_args,
            latent_kwargs=latent_kwargs,
        )
        # If we have graph features, we need to process them and concatenate them with the node features
        if graph_features is not None:
            graph_features = self.graph_features_net(graph_features)
            embeddings = self.aggregate_graph_features(embeddings, graph_features)

        # downstream tasks are given out as is, no softmax or other assumptions
        return self.compute_downstream_tasks(
            embeddings,
            args=downstream_task_args,
            kwargs=downstream_task_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary with keys matching __init__ parameters.
                Must include: encoder_type, encoder_args, encoder_kwargs, downstream_tasks.
                Optional: pooling_layers, aggregate_pooling_type, graph_features_net_type, etc.

        Returns:
            GNNModel: An initialized GNNModel instance.

        Raises:
            RuntimeError: If model creation fails (wraps underlying exceptions).
            jsonschema.ValidationError: If config is invalid.
        """
        try:
            jsonschema.validate(config, cls.schema)
            return cls(
                config["encoder_type"],
                config["downstream_tasks"],
                encoder_args=config.get("encoder_args", None),
                encoder_kwargs=config.get("encoder_kwargs", None),
                pooling_layers=config.get("pooling_layers", None),
                aggregate_pooling_type=config.get("aggregate_pooling_type"),
                aggregate_pooling_args=config.get("aggregate_pooling_args"),
                aggregate_pooling_kwargs=config.get("aggregate_pooling_kwargs"),
                latent_model_type=config.get("latent_model_type"),
                latent_model_args=config.get("latent_model_args"),
                latent_model_kwargs=config.get("latent_model_kwargs"),
                graph_features_net_type=config.get("graph_features_net_type"),
                graph_features_net_args=config.get("graph_features_net_args"),
                graph_features_net_kwargs=config.get("graph_features_net_kwargs"),
                aggregate_graph_features_type=config.get(
                    "aggregate_graph_features_type"
                ),
                aggregate_graph_features_args=config.get(
                    "aggregate_graph_features_args"
                ),
                aggregate_graph_features_kwargs=config.get(
                    "aggregate_graph_features_kwargs"
                ),
                active_tasks=config.get("active_tasks"),
            )
        except Exception as e:
            raise RuntimeError(
                f"Error during creation of GNNModel from config: {e}"
            ) from e

    def save(self, path: str | Path) -> None:
        """Save the model state dictionary to file.

        Args:
            path (str | Path): File path where the model will be saved.
        """

        torch.save(
            self.state_dict(),
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: Dict[str, Any] | None = None,
        args: Sequence[Any] | None = None,
        kwargs: Dict[str, Any] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> "GNNModel":
        """Load a GNNModel from a file saved with the save() method that's defined by the provided config.
        It is assumed that the config used to save the model is the same as the one provided here or defines the
        same model architecture. Therefore, configs should always be saved alongside the model weights.

        Args:
            path (str | Path): Path to the saved model file.
            config (Dict[str, Any] | None): Config for building the model
            args (Sequence[Any] | None): Arguments for building the model if config is not supplied
            kwargs (Dict[str, Any] | None): Keyword argumetns for building the model if config is not supplied
            device (torch.device, optional): Device to load the model onto. Defaults to torch.device("cpu").

        Returns:
            GNNModel: Fully initialized model instance with loaded weights.
        """
        if config is not None:
            model = cls.from_config(config).to(device)
        else:
            model = cls(
                *(args if args is not None else []),
                **(kwargs if kwargs is not None else {}),
            ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model
