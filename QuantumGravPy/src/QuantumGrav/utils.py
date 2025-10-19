from functools import partial
import importlib
import torch
import torch_geometric.nn as tgnn

from typing import Callable

gnn_layers: dict[str, type[torch.nn.Module]] = {
    "gcn": tgnn.conv.GCNConv,
    "gat": tgnn.conv.GATConv,
    "sage": tgnn.conv.SAGEConv,
    "gco": tgnn.conv.GraphConv,
}

normalizer_layers: dict[str, type[torch.nn.Module]] = {
    "identity": torch.nn.Identity,
    "batch_norm": torch.nn.BatchNorm1d,
    "layer_norm": torch.nn.LayerNorm,
}

activation_layers: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}

pooling_layers: dict[str, Callable | type[torch.nn.Module]] = {
    "mean": tgnn.global_mean_pool,
    "max": tgnn.global_max_pool,
    "sum": tgnn.global_add_pool,
    "identity": torch.nn.Identity,
}

pooling_aggregations: dict[str, Callable | type[torch.nn.Module]] = {
    "cat0": torch.cat,
    "cat1": partial(torch.cat, dim=1),
    "identity": torch.nn.Identity,
}

graph_features_aggregations: dict[str, Callable | type[torch.nn.Module]] = {
    "cat0": torch.cat,
    "cat1": partial(torch.cat, dim=1),
    "identity": torch.nn.Identity,
}


def list_registered_pooling_layers() -> list[str]:
    """List all registered pooling layers."""
    return list(pooling_layers.keys())


def list_registered_gnn_layers() -> list[str]:
    """List all registered GNN layers."""
    return list(gnn_layers.keys())


def list_registered_normalizers() -> list[str]:
    """List all registered normalizer layers."""
    return list(normalizer_layers.keys())


def list_registered_activations() -> list[str]:
    """List all registered activation layers."""
    return list(activation_layers.keys())


def list_registered_graph_features_aggregations() -> list[str]:
    """List all registered graph features aggregation functions.

    Returns:
        list[str]: A list of registered graph features aggregation function names.
    """
    return list(graph_features_aggregations.keys())


def list_registered_pooling_aggregations() -> list[str]:
    """List all registered pooling aggregation functions.

    Returns:
        list[str]: A list of registered pooling aggregation function names.
    """
    return list(pooling_aggregations.keys())


def register_pooling_layer(
    pooling_layer_name: str, pooling_layer: type[torch.nn.Module] | Callable
) -> None:
    """Register a pooling layer with the module

    Args:
        pooling_layer_name (str): The name of the pooling layer.
        pooling_layer (torch.nn.Module): The pooling layer to register.
    """
    if pooling_layer_name in pooling_layers:
        raise ValueError(f"Pooling layer '{pooling_layer_name}' is already registered.")
    pooling_layers[pooling_layer_name] = pooling_layer


def register_gnn_layer(gnn_layer_name: str, gnn_layer: type[torch.nn.Module]) -> None:
    """Register a GNN layer with the module

    Args:
        gnn_layer_name (str): The name of the GNN layer.
        gnn_layer (type[torch.nn.Module]): The GNN layer to register.
    """
    if gnn_layer_name in gnn_layers:
        raise ValueError(f"GNN layer '{gnn_layer_name}' is already registered.")
    gnn_layers[gnn_layer_name] = gnn_layer


def register_normalizer(
    normalizer_name: str, normalizer_layer: type[torch.nn.Module]
) -> None:
    """Register a normalizer layer with the module

    Args:
        normalizer_name (str): The name of the normalizer.
        normalizer_layer (type[torch.nn.Module]): The normalizer layer to register.

    Raises:
        ValueError: If the normalizer layer is already registered.
    """
    if normalizer_name in normalizer_layers:
        raise ValueError(f"Normalizer '{normalizer_name}' is already registered.")
    normalizer_layers[normalizer_name] = normalizer_layer


def register_activation(
    activation_name: str, activation_layer: type[torch.nn.Module]
) -> None:
    """Register an activation layer with the module

    Args:
        activation_name (str): The name of the activation layer.
        activation_layer (type[torch.nn.Module]): The activation layer to register.

    Raises:
        ValueError: If the activation layer is already registered.
    """
    if activation_name in activation_layers:
        raise ValueError(f"Activation '{activation_name}' is already registered.")
    activation_layers[activation_name] = activation_layer


def register_graph_features_aggregation(
    aggregation_name: str, aggregation_function: type[torch.nn.Module] | Callable
) -> None:
    """Register a graph features aggregation function with the module

    Args:
        aggregation_name (str): The name of the graph features aggregation function.
        aggregation_function (Callable): The graph features aggregation function to register.

    Raises:
        ValueError: If the graph features aggregation function is already registered.
    """
    if aggregation_name in graph_features_aggregations:
        raise ValueError(
            f"Graph features aggregation '{aggregation_name}' is already registered."
        )
    graph_features_aggregations[aggregation_name] = aggregation_function


def register_pooling_aggregation(
    aggregation_name: str, aggregation_function: Callable
) -> None:
    """Register a pooling aggregation function with the module

    Args:
        aggregation_name (str): The name of the pooling aggregation function.
        aggregation_function (Callable): The pooling aggregation function to register.

    Raises:
        ValueError: If the pooling aggregation function is already registered.
    """
    if aggregation_name in pooling_aggregations:
        raise ValueError(
            f"Pooling aggregation '{aggregation_name}' is already registered."
        )
    pooling_aggregations[aggregation_name] = aggregation_function


def get_registered_pooling_layer(name: str) -> type[torch.nn.Module] | Callable | None:
    """Get a registered pooling layer by name.

    Args:
        name (str): The name of the pooling layer.

    Returns:
        type[torch.nn.Module] | Callable | None: The registered pooling layer named `name`, or None if not found.
    """
    return pooling_layers[name] if name in pooling_layers else None


def get_registered_gnn_layer(name: str) -> type[torch.nn.Module] | None:
    """Get a registered GNN layer by name.
    Args:
        name (str): The name of the GNN layer.

    Returns:
        type[torch.nn.Module] | None: The registered GNN layer named `name`, or None if not found.
    """
    return gnn_layers[name] if name in gnn_layers else None


def get_registered_normalizer(name: str) -> type[torch.nn.Module] | None:
    """Get a registered normalizer layer by name.

    Args:
        name (str): The name of the normalizer layer.

    Returns:
        type[torch.nn.Module]| None: The registered normalizer layer named `name`, or None if not found.
    """
    return normalizer_layers[name] if name in normalizer_layers else None


def get_registered_activation(name: str) -> type[torch.nn.Module] | None:
    """Get a registered activation layer by name.

    Args:
        name (str): The name of the activation layer.

    Returns:
        type[torch.nn.Module] | None: The registered activation layer named `name`, or None if not found.
    """
    return activation_layers[name] if name in activation_layers else None


def get_pooling_aggregation(name: str) -> type[torch.nn.Module] | Callable | None:
    """Get a registered pooling aggregation function by name.

    Args:
        name (str): The name of the pooling aggregation function.

    Returns:
        type[torch.nn.Module] | Callable: Function to aggregate pooling layer outputs.
    """
    return pooling_aggregations[name] if name in pooling_aggregations else None


def get_graph_features_aggregation(
    name: str,
) -> type[torch.nn.Module] | Callable | None:
    """Get a registered graph features aggregation function by name.

    Args:
        name (str): The name of the graph features aggregation function.

    Returns:
        type[torch.nn.Module] | Callable | None: Function to aggregate graph features outputs, or None if not found.
    """
    return (
        graph_features_aggregations[name]
        if name in graph_features_aggregations
        else None
    )


def verify_config_node(cfg) -> bool:
    """Verify that a config node has the required keys.

    Args:
        cfg (dict): The config node to verify.

    Returns:
        bool: True if the config node is valid, False otherwise.
    """
    required_keys = {"type", "args", "kwargs"}
    if not isinstance(cfg, dict):
        return False
    if not required_keys.issubset(cfg.keys()):
        return False
    if not isinstance(cfg["args"], list):
        return False
    if not isinstance(cfg["kwargs"], dict):
        return False
    return True


def import_and_get(modulename: str, objectname: str) -> type | None:
    """Import a module and get an object from it.

    Args:
        modulename (str): The name of the module to import.
        objectname (str): The name of the object to get from the module.
    Returns:
        type | None: The object from the module.
    """

    try:
        module = importlib.import_module(modulename)
    except Exception as e:
        raise ValueError(f"Importing module {modulename} unsuccessful") from e

    try:
        tpe = getattr(module, objectname)
    except Exception as e:
        raise ValueError(f"Could not load name {objectname} from {modulename}") from e
    
    return tpe
