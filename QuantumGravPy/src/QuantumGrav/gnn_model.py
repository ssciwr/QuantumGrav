from typing import Any, Callable, Mapping, Sequence, Dict, Tuple
from pathlib import Path
from inspect import isclass
import jsonschema

import torch

from .base import BaseModel


class ModuleWrapper(torch.nn.Module):
    """Wrapper to make pooling functions compatible with ModuleList."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def get_fn(self) -> Callable:
        return self.fn


class GNNModel(BaseModel):
    """Torch module for the full GCN model, which consists of a GCN backbone, a set of downstream tasks, and a set of  pooling layers, augmented with optional graph features network. Additionally, aggregation methods for the pooling and graph layers can be supplied
    Args:
        torch.nn.Module: base class
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GNNModel Configuration",
        "type": "object",
        "properties": {
            "encoder_type": {
                "description": "name of the encoder model",
            },
            "encoder_args": {
                "type": "array",
                "description": "Positional arguments for the encoder",
                "items": {},
            },
            "encoder_kwargs": {
                "type": "object",
                "description": "keyword arguments for the encoder",
                "additionalProperties": True,
            },
            "downstream_tasks": {
                "type": "object",
                "description": "Dictionary of downstream tasks",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": [
                        {
                            "description": "Module type name",
                        },
                        {
                            "type": "array",
                            "description": "Positional arguments",
                            "items": {},
                        },
                        {
                            "type": "object",
                            "description": "Keyword arguments",
                            "additionalProperties": True,
                        },
                    ],
                    "required": ["type"],
                    "additionalItems": True,  # allows extra elements beyond the first 3
                },
            },
            "pooling_layers": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": [
                        {
                            "description": "Module type name",
                        },
                        {
                            "type": "array",
                            "description": "Positional arguments",
                            "items": {},
                        },
                        {
                            "type": "object",
                            "description": "Keyword arguments",
                            "additionalProperties": True,
                        },
                    ],
                    "additionalItems": True,  # allows extra elements beyond the first 3
                    "required": ["type"],
                },
            },
            "aggregate_pooling_type": {
                "description": "class- or function name of the aggregate pooling layer",
            },
            "aggregate_pooling_args": {
                "type": "array",
                "description": "Positional arguments for the aggregate_pooling layers",
                "items": {},
            },
            "aggregate_pooling_kwargs": {
                "type": "object",
                "description": "keyword arguments for the pooling aggregation",
                "additionalProperties": True,
            },
            "graph_features_net_type": {
                "description": "class name of the graph features net",
            },
            "graph_features_net_args": {
                "type": "array",
                "description": "Positional arguments for the graph_features_net",
                "items": {},
            },
            "graph_features_net_kwargs": {
                "type": "object",
                "description": "keyword arguments for the graph features net",
                "additionalProperties": True,
            },
            "aggregate_graph_features_type": {
                "description": "class or function name of the graph feature aggregation",
            },
            "aggregate_graph_features_args": {
                "type": "array",
                "description": "Positional arguments for the aggregate_graph_features layers",
                "items": {},
            },
            "aggregate_graph_features_kwargs": {
                "type": "object",
                "description": "keyword arguments for the aggregate graph features note",
                "additionalProperties": True,
            },
            "active_tasks": {
                "type": "object",
                "description": "Dictionary for active tasks",
                "additionaProperties": {"type": "boolean"},
            },
        },
        "required": [
            "encoder_type",
            "encoder_args",
            "encoder_kwargs",
            "downstream_tasks",
            "pooling_layers",
            "aggregate_pooling_type",
            "active_tasks",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        encoder_type: type[torch.nn.Module],
        encoder_args: Sequence[Any],
        encoder_kwargs: Dict[str, Any],
        downstream_tasks: Dict[
            str, Tuple[type[torch.nn.Module], Sequence[Any], Dict[str, Any]]
        ],
        pooling_layers: Sequence[
            Tuple[type[torch.nn.Module] | Callable, Sequence[Any], Dict[str, Any]]
        ]
        | None = None,
        aggregate_pooling_type: type[torch.nn.Module] | None = None,
        aggregate_pooling_args: Sequence[Any] | None = None,
        aggregate_pooling_kwargs: Dict[str, Any] | None = None,
        graph_features_net_type: type[torch.nn.Module] | None = None,
        graph_features_net_args: Sequence[Any] | None = None,
        graph_features_net_kwargs: Dict[str, Any] | None = None,
        aggregate_graph_features_type: type[torch.nn.Module] | None = None,
        aggregate_graph_features_args: Sequence[Any] | None = None,
        aggregate_graph_features_kwargs: Dict[str, Any] | None = None,
        active_tasks: Dict[Any, bool] | None = None,
    ):
        """Instantiate a new GNNModel consisting of an encoder, a set of pooling functions or torch.nn.Modules and an aggregations method for them, as well as an optional net for integrating graph level features and its accompanying aggregation method. Finally a set of downstream tasks can be added, which can be activated
        or deactivated with the passed 'active_tasks'.

        Args:
            encoder_type (type[torch.nn.Module]): Type name of the encoder name
            encoder_args (Sequence[Any]): Positional arguments of the encoder
            encoder_kwargs (Dict[str, Any]): Keyword arguments of the encoder
            downstream_tasks (Dict[ Any, Tuple[type[torch.nn.Module], Sequence[Any], Dict[str, Any]] ]): Dictionary of torch module subclass name and accompanying args and kwargs for its instantiation that represents the downstream tasks. Each tuple element must look like this:
                - first: (type[torch.nn.Module]): The class of the downstream model to instantiate.
                - second: (Sequence[Any]): Positional arguments to pass to the model constructor.
                - third: (dict[str, Any]): Keyword arguments to pass to the model constructor.
            pooling_layers (Sequence[Tuple[type[torch.nn.Module] | Callable, Sequence[Any], Dict[str, Any]]] | None, optional): Sequence of tuples containing class or function name, args and kwargs for the pooling layers to be used in this order:
                - first: (type[torch.nn.Module] | Callable): The class of the pooling layer to instantiate or a function used for that purpose.
                - second: (Sequence[Any]): Positional arguments to pass to the pooling layer constructor.
                - third: (dict[str, Any]): Keyword arguments to pass to the pooling layer constructor.
            Defaults to None.
            aggregate_pooling_type (type[torch.nn.Module] | None, optional): torch subclass name or callable for aggregating the pooling layers' outputs. Defaults to None.
            aggregate_pooling_args (Sequence[Any] | None, optional): Positional arguments for the pooling layer aggregation. Defaults to None.
            aggregate_pooling_kwargs (Dict[str, Any] | None, optional): Keyword arguments for the pooling layer aggregation. Defaults to None.
            graph_features_net_type (type[torch.nn.Module] | None, optional): Name of the graph features net. Defaults to None.
            graph_features_net_args (Sequence[Any] | None, optional): Positional arguments for the graph features. Defaults to None.
            graph_features_net_kwargs (Dict[str, Any] | None, optional): Keyword arguments for the graph features net. Defaults to None.
            aggregate_graph_features_type (type[torch.nn.Module] | None, optional): Name of the class for callable used to merge graph features with graph embeddings from the encoder and pooling layers. Defaults to None.
            aggregate_graph_features_args (Sequence[Any] | None, optional): Positional arguments of the graph feature aggregation. Defaults to None.
            aggregate_graph_features_kwargs (Dict[str, Any] | None, optional): Keyword arguments of the graph feature aggregations.. Defaults to None.
            active_tasks (Dict[Any, bool] | None, optional): Dictionary indicating which tasks should be active after instantiation. Defaults to None.

        Raises:
            TypeError: When aggregate_pooling_type is not a torch module or callable
            TypeError: When aggregate_graph_features_type is not a torch module or callable

        Returns:
            GNNModel: A new GNNModel instance
        """
        super().__init__()

        # store stuff that will become the config for to_config
        self.encoder_specs = {
            "type": encoder_type,
            "args": encoder_args,
            "kwargs": encoder_kwargs,
        }
        self.downstream_tasks_specs = downstream_tasks
        self.pooling_layers_specs = pooling_layers
        self.aggregate_pooling_specs = {
            "type": aggregate_pooling_type,
            "args": aggregate_pooling_args,
            "kwargs": aggregate_pooling_kwargs,
        }
        self.graph_features_net_specs = {
            "type": graph_features_net_type,
            "args": graph_features_net_args,
            "kwargs": graph_features_net_kwargs,
        }
        self.aggregate_graph_features_specs = {
            "type": aggregate_graph_features_type,
            "args": aggregate_graph_features_args,
            "kwargs": aggregate_graph_features_kwargs,
        }
        self.active_tasks_specs = active_tasks

        # set up encoder
        self.encoder: torch.nn.Module = encoder_type(*encoder_args, **encoder_kwargs)

        # set up pooling layers
        if pooling_layers is not None:
            if len(pooling_layers) == 0:
                raise ValueError("At least one pooling layer must be provided.")

            self.pooling_layers = torch.nn.ModuleList(
                [
                    ModuleWrapper(ptype)
                    if callable(ptype)
                    else ptype(
                        *(args if args is not None else []),
                        **(kwargs if kwargs is not None else {}),
                    )
                    for (ptype, args, kwargs) in pooling_layers
                ]
            )

        # set up aggregation of pooling layers
        if aggregate_pooling_type is not None:
            if isclass(aggregate_pooling_type) and issubclass(
                aggregate_pooling_type, torch.nn.Module
            ):
                self.aggregate_pooling: torch.nn.Module = aggregate_pooling_type(
                    *(aggregate_pooling_args if aggregate_pooling_args else []),
                    **(aggregate_pooling_kwargs if aggregate_pooling_kwargs else {}),
                )
            elif callable(aggregate_pooling_type):
                self.aggregate_pooling: torch.nn.Module = ModuleWrapper(
                    aggregate_pooling_type
                )
            else:
                raise TypeError(
                    "Error, aggregate_pooling_type must be a callable or a subclass of torch.nn.Module"
                )
        else:
            self.aggregate_pooling = None

        # set up graph_features network if it's provided
        if graph_features_net_type is not None:
            self.graph_features_net: torch.nn.Module = graph_features_net_type(
                *(graph_features_net_args if graph_features_net_args else []),
                **(graph_features_net_kwargs if graph_features_net_kwargs else {}),
            )
        else:
            self.graph_features_net = None

        # set up aggregation of graph features
        if aggregate_graph_features_type is not None:
            if isclass(aggregate_graph_features_type) and issubclass(
                aggregate_graph_features_type, torch.nn.Module
            ):
                self.aggregate_graph_features: torch.nn.Module = (
                    aggregate_graph_features_type(
                        *(
                            aggregate_graph_features_args
                            if aggregate_graph_features_args
                            else []
                        ),
                        **(
                            aggregate_graph_features_kwargs
                            if aggregate_graph_features_kwargs
                            else {}
                        ),
                    )
                )
            elif callable(aggregate_graph_features_type):
                self.aggregate_graph_features: torch.nn.Module = ModuleWrapper(
                    aggregate_graph_features_type
                )
            else:
                raise TypeError(
                    "Error, aggregate_graph_features_type must be a callable or a subclass of torch.nn.Module"
                )
        else:
            self.aggregate_graph_features = None

        # set up downstream tasks
        self.downstream_tasks = torch.nn.ModuleDict(
            {
                key: taskspecs[0](
                    *(taskspecs[1] if taskspecs[1] else []),
                    **(taskspecs[2] if taskspecs[2] else {}),
                )
                for key, taskspecs in downstream_tasks.items()
            }
        )

        self.active_tasks: Dict[Any, bool] = (
            active_tasks
            if active_tasks is not None
            else {key: True for key in downstream_tasks.keys()}
        )

    def set_task_active(self, key: Any) -> None:
        """Set a downstream task as active.

        Args:
            key (Any): Index of the downstream task to activate.
        """

        if key not in self.active_tasks:
            raise KeyError("Invalid task index.")

        self.active_tasks[key] = True

    def set_task_inactive(self, key: Any) -> None:
        """Set a downstream task as inactive.

        Args:
            i (Any): Index of the downstream task to deactivate.
        """

        if key not in self.active_tasks:
            raise KeyError("Invalid task index.")

        self.active_tasks[key] = False

    def get_graph_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor | None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Get embeddings from the GCN model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Graph connectivity information.
            edge_weight (torch.Tensor): Edge weights.
            batch (torch.Tensor | None): batch indicator tensor
            *args: additional args for the encoder
            **kwargs: additional kwargs for the encoder

        Returns:
            torch.Tensor: Embedding vector for the graph features.
        """
        embeddings = self.encoder(x, edge_index, edge_weight, **kwargs)

        # Pool to graph-level if pooling is configured
        if self.pooling_layers is not None and self.aggregate_pooling is not None:
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            pooled = [pool(embeddings, batch) for pool in self.pooling_layers]
            embeddings = self.aggregate_pooling(pooled)

        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        graph_features: torch.Tensor | None = None,
        **kwargs,
    ) -> Mapping[int, torch.Tensor]:
        """Compute the full model output

        Args:
            x (torch.Tensor): _description_
            edge_index (torch.Tensor): _description_
            batch (torch.Tensor | None, optional): _description_. Defaults to None.
            graph_features (torch.Tensor | None, optional): _description_. Defaults to None.
            kwargs: further kwargs. can contain some or all of:
            {
                "encoder_args": [arg1, arg2],
                "encoder_kwargs": {
                    "kwarg1": kwargvalue1,
                    "kwarg2": kwargvalue2,
                },

            }
        Returns:
            Mapping[int, torch.Tensor]: _description_
        """

        embeddings = self.get_graph_embeddings(
            x,
            edge_index,
            edge_weight,
            batch,
            *(kwargs.get("encoder_args", [])),
            **(kwargs.get("encoder_kwargs", {})),
        )

        if self.graph_features_net is not None:
            embeddings = self.graph_features_net(embeddings, graph_features)

        # Compute downstream tasks - pass batch to tasks that might need it
        output = {}
        for key, task in self.downstream_tasks.items():
            if self.active_tasks[key]:
                # Tasks get embeddings and can use batch from kwargs if needed
                output[key] = task(
                    embeddings,
                    *kwargs.get("downstream_args", {}).get(key, []),
                    **kwargs.get("downstream_kwargs", {}).get(key, {}),
                )

        return output

    @classmethod
    def verify_config(cls, config: Dict[str, Any]) -> bool:
        """_summary_

        Args:
            config (Dict[str, Any]): _description_

        Returns:
            bool: _description_
        """
        jsonschema.validate(config, schema=cls.schema)
        return True

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """
        # throws if verification fails
        cls.verify_config(config)

        # return the model
        return cls(
            config["encoder_type"],
            config["encoder_args"],
            config["encoder_kwargs"],
            downstream_tasks=config["downstream_tasks"],
            pooling_layers=config.get("pooling_layers", None),
            aggregate_pooling_type=config.get("aggregate_pooling_type", None),
            aggregate_pooling_args=config.get("aggregate_pooling_args", None),
            aggregate_pooling_kwargs=config.get("aggregate_pooling_kwargs", None),
            graph_features_net_type=config.get("graph_features_net_type", None),
            graph_features_net_args=config.get("graph_features_net_args", None),
            graph_features_net_kwargs=config.get("graph_features_net_kwargs", None),
            aggregate_graph_features_type=config.get(
                "aggregate_graph_features_type", None
            ),
            aggregate_graph_features_args=config.get(
                "aggregate_graph_features_args", None
            ),
            aggregate_graph_features_kwargs=config.get(
                "aggregate_graph_features_kwargs", None
            ),
            active_tasks=config.get("active_tasks", None),
        )

    def to_config(self) -> Dict[str, Any]:
        """Serialize the model to a config

        Returns:
            Dict[str, Any]: Config representation of the caller
        """
        config: Dict[str, Any] = {}

        config["encoder"] = {
            "type": self.encoder_specs["type"],
            "args": self.encoder_specs["args"],
            "kwargs": self.encoder_specs["kwargs"],
        }
        config["downstream_tasks"] = self.downstream_tasks_specs
        config["pooling_layers"] = self.pooling_layers_specs
        config["aggregate_pooling"] = {
            "type": self.aggregate_pooling_specs["type"],
            "args": self.aggregate_pooling_specs["args"],
            "kwargs": self.aggregate_pooling_specs["kwargs"],
        }
        config["graph_features_net"] = {
            "type": self.graph_features_net_specs["type"],
            "args": self.graph_features_net_specs["args"],
            "kwargs": self.graph_features_net_specs["kwargs"],
        }
        config["aggregate_graph_features"] = {
            "type": self.aggregate_graph_features_specs["type"],
            "args": self.aggregate_graph_features_specs["args"],
            "kwargs": self.aggregate_graph_features_specs["kwargs"],
        }
        config["active_tasks"] = self.active_tasks_specs

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
