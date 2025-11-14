from typing import Any, Callable, Mapping, Sequence, Dict, Tuple
from pathlib import Path
import torch
from . import utils
from . import sequential_model as QGGNN
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
    """Torch module for the full GCN model, which consists of a GCN backbone, a set of downstream tasks, and a pooling layer, augmented with optional graph features network.
    Args:
        torch.nn.Module: base class
    """

    def __init__(
        self,
        encoder_name: type[torch.nn.Module],
        encoder_args: Sequence[Any],
        encoder_kwargs: Dict[str, Any],
        downstream_tasks: Dict[
            Any, Tuple[type[torch.nn.Module], Sequence[Any], Dict[str, Any]]
        ],
        pooling_layers: Sequence[Dict[str, Any]] | None = None,
        aggregate_pooling_name: type[torch.nn.Module] | None = None,
        aggregate_pooling_args: Sequence[Any] | None = None,
        aggregate_pooling_kwargs: Dict[str, Any] | None = None,
        graph_features_net_name: type[torch.nn.Module] | None = None,
        graph_features_net_args: Sequence[Any] | None = None,
        graph_features_net_kwargs: Dict[str, Any] | None = None,
        aggregate_graph_features_name: type[torch.nn.Module] | None = None,
        aggregate_graph_features_args: Sequence[Any] | None = None,
        aggregate_graph_features_kwargs: Dict[str, Any] | None = None,
        active_tasks: Dict[Any, bool] | None = None,
    ):
        """_summary_

        Args:
            encoder_name (type[torch.nn.Module]): _description_
            encoder_args (Sequence[Any]): _description_
            encoder_kwargs (Dict[str, Any]): _description_
            downstream_tasks (Dict[ Any, Tuple[type[torch.nn.Module], Sequence[Any], Dict[str, Any]] ]): _description_
            pooling_layers (Sequence[Dict[str, Any]] | None, optional): _description_. Defaults to None.
            aggregate_pooling_name (type[torch.nn.Module] | None, optional): _description_. Defaults to None.
            aggregate_pooling_args (Sequence[Any] | None, optional): _description_. Defaults to None.
            aggregate_pooling_kwargs (Dict[str, Any] | None, optional): _description_. Defaults to None.
            graph_features_net_name (type[torch.nn.Module] | None, optional): _description_. Defaults to None.
            graph_features_net_args (Sequence[Any] | None, optional): _description_. Defaults to None.
            graph_features_net_kwargs (Dict[str, Any] | None, optional): _description_. Defaults to None.
            aggregate_graph_features_name (type[torch.nn.Module] | None, optional): _description_. Defaults to None.
            aggregate_graph_features_args (Sequence[Any] | None, optional): _description_. Defaults to None.
            aggregate_graph_features_kwargs (Dict[str, Any] | None, optional): _description_. Defaults to None.
            active_tasks (Dict[Any, bool] | None, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()

        # FIXME: Some parts here can be modules or functions, so have to be wrapped in the ModuleWrapper 
        # store stuff that will become the config for to_config
        self.encoder_specs = {
            "type": encoder_name,
            "args": encoder_args,
            "kwargs": encoder_kwargs,
        }
        self.downstream_tasks_specs = downstream_tasks
        self.pooling_layers_specs = pooling_layers
        self.aggregate_pooling_specs = {
            "type": aggregate_pooling_name,
            "args": aggregate_pooling_args,
            "kwargs": aggregate_pooling_kwargs,
        }
        self.graph_features_net_specs = {
            "type": graph_features_net_name,
            "args": graph_features_net_args,
            "kwargs": graph_features_net_kwargs,
        }
        self.aggregate_graph_features_specs = {
            "type": aggregate_graph_features_name,
            "args": aggregate_graph_features_args,
            "kwargs": aggregate_graph_features_kwargs,
        }
        self.active_tasks_specs = active_tasks

        # set up encoder
        self.encoder: torch.nn.Module = encoder_name(*encoder_args, **encoder_kwargs)

        # set up pooling layers
        if pooling_layers is not None:
            if len(pooling_layers) == 0:
                raise ValueError("At least one pooling layer must be provided.")

            self.pooling_layers = torch.nn.ModuleList(
                [
                    p["type"](*p.get("args", []), **p.get("kwargs", {}))
                    for p in pooling_layers
                ]
            )

        # FIXME: types can be functions, so have to be wrapped in the ModuleWrapper
        # set up aggregation of pooling layers
        if aggregate_pooling_name is not None:
            self.aggregate_pooling: torch.nn.Module = aggregate_pooling_name(
                *(aggregate_pooling_args if aggregate_pooling_args else []),
                **(aggregate_pooling_kwargs if aggregate_pooling_kwargs else {}),
            )

        # set up graph_features network if it's provided
        if graph_features_net_name is not None:
            self.graph_features_net: torch.nn.Module = graph_features_net_name(
                *(graph_features_net_args if graph_features_net_args else []),
                **(graph_features_net_kwargs if graph_features_net_kwargs else {}),
            )

        # set up aggregation of graph features
        if aggregate_graph_features_name is not None:
            self.aggregate_graph_features: torch.nn.Module = (
                aggregate_graph_features_name(
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

        self.downstream_tasks = torch.nn.ModuleDict(
            {
                key: taskspecs[0](*taskspecs[1], **taskspecs[2])
                for key, taskspecs in downstream_tasks.items()
            }
        )

        self.active_tasks: Dict[Any, bool] = (
            active_tasks
            if active_tasks is not None
            else {i: True for i in range(len(downstream_tasks))}
        )

    def set_task_active(self, key: Any) -> None:
        """Set a downstream task as active.

        Args:
            key (Any): Index of the downstream task to activate.
        """

        if key not in self.active_tasks:
            raise ValueError("Invalid task index.")

        self.active_tasks[key] = True

    def set_task_inactive(self, key: Any) -> None:
        """Set a downstream task as inactive.

        Args:
            i (Any): Index of the downstream task to deactivate.
        """

        if key not in self.active_tasks:
            raise ValueError("Invalid task index.")

        self.active_tasks[key] = False

    def get_graph_embeddings(
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
        embeddings = self.encoder(x, edge_index, **(gcn_kwargs if gcn_kwargs else {}))

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
        downstream_task_args_kwargs: Dict[
            Any, Dict[str, Sequence[Any] | Dict[str, Any]]
        ]
        | None = None,
    ) -> dict[int, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            downstream_task_args_kwargs (Dict[ Any, Dict[Any, Sequence[Any]  |  Dict[str, Any]] ] | None, optional): _description_. Defaults to None.

        Returns:
            dict[int, torch.Tensor]: _description_
        """

        output = {}
        if downstream_task_args_kwargs is None:
            task_ak: Dict[Any, Dict[str, Sequence[Any] | Dict[str, Any]]] = {}
        else:
            task_ak = downstream_task_args_kwargs

        for key, task in self.downstream_tasks.items():
            if self.active_tasks[key]:
                current = task_ak.get(key, {})
                task_args: Sequence[Any] = (
                    current.get("args", []) if "args" in current else []
                )
                task_kwargs: Dict[str, Any] = (
                    current.get("kwargs", {}) if "kwargs" in current else {}
                )
                res = task(x, *task_args, **task_kwargs)
                output[key] = res

        return output

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        gcn_args: Sequence[Any] | None = None,
        gcn_kwargs: Dict[str, Any] | None = None,
        graph_features_args: Sequence[Any] | None = None,
        graph_features_kwargs: Dict[str, Any] | None = None,
        downstream_task_args_kwargs: Dict[
            Any, Dict[str, Sequence[Any] | Dict[str, Any]]
        ]
        | None = None,
    ) -> torch.Tensor | Mapping[int, torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            edge_index (torch.Tensor): _description_
            batch (torch.Tensor | None, optional): _description_. Defaults to None.
            gcn_args (Sequence[Any] | None, optional): _description_. Defaults to None.
            gcn_kwargs (Dict[str, Any] | None, optional): _description_. Defaults to None.
            graph_features_args (Sequence[Any] | None, optional): _description_. Defaults to None.
            graph_features_kwargs (Dict[str, Any] | None, optional): _description_. Defaults to None.
            downstream_task_args_kwargs (Dict[ Any, Dict[str, Sequence[Any]  |  Dict[str, Any]] ] | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor | Mapping[int, torch.Tensor]: _description_
        """
        # FIXME: this will probably not work, make sure that args/kwargs shit works as it should

        # apply the GCN backbone to the node features
        embeddings = self.get_graph_embeddings(
            x,
            edge_index,
            *(gcn_args if gcn_args else []),
            **(gcn_kwargs if gcn_kwargs else {}),
        )

        # If we have graph features, we need to process them and concatenate them with the node features
        if self.graph_features_net:
            graph_features = self.graph_features_net(
                *(graph_features_args if graph_features_args else []),
                **(graph_features_kwargs if graph_features_kwargs else {}),
            )
            embeddings = self.aggregate_graph_features(
                embeddings,
                graph_features,
                **(graph_features_kwargs if graph_features_kwargs else {}),
            )

        # downstream tasks are given out as is, no softmax or other assumptions
        return self.compute_downstream_tasks(
            embeddings,
            downstream_task_args_kwargs=downstream_task_args_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "GNNModel":
        """Create a GNNModel from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the model.

        Returns:
            GNNModel: An instance of GNNModel.
        """
        # return the model
        return cls(
            config["encoder"]["type"],
            config["encoder"]["args"],
            config["encoder"]["kwargs"],
            downstream_tasks={
                key: (
                    task_cfg["type"],
                    task_cfg.get("args", []),
                    task_cfg.get("kwargs", {}),
                )
                for (key, task_cfg) in config["downstream_tasks"].items()
            },
            pooling_layers=config.get("pooling_layers", None),
            aggregate_pooling_name=config["aggregate_pooling"]["type"],
            aggregate_pooling_args=config["aggregate_pooling"].get("args", None),
            aggregate_pooling_kwargs=config["aggregate_pooling"].get("kwargs", None),
            graph_features_net_name=config["graph_features_net"].get("type", None),
            graph_features_net_args=config["graph_features_net"].get("args", None),
            graph_features_net_kwargs=config["graph_features_net"].get("kwargs", None),
            aggregate_graph_features_name=config["aggregate_graph_features"].get(
                "type", None
            ),
            aggregate_graph_features_args=config["aggregate_graph_features"].get(
                "args", None
            ),
            aggregate_graph_features_kwargs=config["aggregate_graph_features"].get(
                "kwargs", None
            ),
            active_tasks=config.get("active_tasks", None),
        )

    def to_config(self) -> Dict[str, Any]:
        """Serialize the model to a config

        Returns:
            Dict[str, Any]: _description_
        """
        config: Dict[str, Any] = {
            "encoder": {
                "type": self.encoder_specs["type"],
                "args": self.encoder_specs["args"],
                "kwargs": self.encoder_specs["kwargs"],
            },
            "downstream_tasks": {
                key: {
                    "type": taskspecs[0],
                    "args": taskspecs[1],
                    "kwargs": taskspecs[2],
                }
                for (key, taskspecs) in self.downstream_tasks.items()
            },
        }

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
