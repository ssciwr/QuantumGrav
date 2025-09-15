import torch
import torch_geometric.nn as tgnn

from typing import Any

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

pooling_layers: dict[str, Any] = {
    "mean": tgnn.global_mean_pool,
    "max": tgnn.global_max_pool,
    "sum": tgnn.global_add_pool,
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


def register_pooling_layer(
    pooling_layer_name: str, pooling_layer: torch.nn.Module
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


def get_registered_pooling_layer(name: str) -> torch.nn.Module | None:
    """Get a registered pooling layer by name.

    Args:
        name (str): The name of the pooling layer.

    Returns:
        torch.nn.Module | None: The registered pooling layer named `name`, or None if not found.
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
