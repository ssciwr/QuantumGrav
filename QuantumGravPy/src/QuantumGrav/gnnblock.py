# torch
import torch
import torch_geometric.nn as tgnn

# quality of life
from typing import Any

gnn_layers: dict[str, torch.nn.Module] = {
    "gcn": tgnn.conv.GCNConv,
    "gat": tgnn.conv.GATConv,
    "sage": tgnn.conv.SAGEConv,
    "graph": tgnn.conv.GraphConv,
}

normalizer_layers: dict[str, torch.nn.Module] = {
    "identity": torch.nn.Identity,
    "batch_norm": torch.nn.BatchNorm1d,
    "layer_norm": torch.nn.LayerNorm,
}

activation_layers: dict[str, torch.nn.Module] = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}


def register_gnn_layer(gcn_layer_name: str, gcn_layer: torch.nn.Module) -> None:
    """Register a GNN layer with the module

    Args:
        gcn_layer_name (str): The name of the GNN layer.
        gcn_layer (torch.nn.Module): The GNN layer to register.
    """
    if gcn_layer_name in gnn_layers:
        raise ValueError(f"GNN layer '{gcn_layer_name}' is already registered.")
    gnn_layers[gcn_layer_name] = gcn_layer


def register_normalizer(
    normalizer_name: str, normalizer_layer: torch.nn.Module
) -> None:
    """Register a normalizer layer with the module

    Args:
        normalizer_name (str): The name of the normalizer.
        normalizer_layer (torch.nn.Module): The normalizer layer to register.

    Raises:
        ValueError: If the normalizer layer is already registered.
    """
    if normalizer_name in normalizer_layers:
        raise ValueError(f"Normalizer '{normalizer_name}' is already registered.")
    normalizer_layers[normalizer_name] = normalizer_layer


def register_activation(
    activation_name: str, activation_layer: torch.nn.Module
) -> None:
    """Register an activation layer with the module

    Args:
        activation_name (str): The name of the activation layer.
        activation_layer (torch.nn.Module): The activation layer to register.

    Raises:
        ValueError: If the activation layer is already registered.
    """
    if activation_name in activation_layers:
        raise ValueError(f"Activation '{activation_name}' is already registered.")
    activation_layers[activation_name] = activation_layer


def get_registered_gnn_layer(name: str) -> dict[str, torch.nn.Module]:
    """Get the registered GNN layers.

    Returns:
        dict[str, torch.nn.Module]: The registered GNN layers.
    """
    return gnn_layers[name] if name in gnn_layers else None


def get_registered_normalizer(name: str) -> dict[str, torch.nn.Module]:
    """Get the registered normalizer layers.

    Returns:
        dict[str, torch.nn.Module]: The registered normalizer layers.
    """
    return normalizer_layers[name] if name in normalizer_layers else None


def get_registered_activation(name: str) -> dict[str, torch.nn.Module]:
    """Get the registered activation layers.

    Returns:
        dict[str, torch.nn.Module]: The registered activation layers.
    """
    return activation_layers[name] if name in activation_layers else None


class GNNBlock(torch.nn.Module):
    """Graph Neural Network Block. Consists of a GNN layer, a normalizer, an activation function,
    and a residual connection. The gcn-layer is applied first, followed by the normalizer and activation function. The result is then projected into the input space using a linear layer and added to the original input (residual connection). Finally, dropout is applied for regularization.

    Args:
        torch (torch.nn.Module): Derives from torch.nn.Module to create a neural network block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        gcn_type: torch.nn.Module = tgnn.conv.GCNConv,
        normalizer: torch.nn.Module = torch.nn.Identity,
        activation: torch.nn.Module = torch.nn.ReLU,
        gcn_args: list[Any] = None,
        gcn_kwargs: dict[str, Any] = None,
        norm_args: list[Any] = None,
        norm_kwargs: dict[str, Any] = None,
        activation_args: list[Any] = None,
        activation_kwargs: dict[str, Any] = None,
    ):
        """Create a GNNBlock instance.

        Args:
            in_channels (int): The dimensions of the input features.
            out_channels (int): The dimensions of the output features.
            dropout (float, optional): The dropout probability. Defaults to 0.3.
            gcn_type (torch.nn.Module, optional): The type of GCN-layer to use. Defaults to tgnn.conv.GCNConv.
            normalizer (torch.nn.Module, optional): The normalizer layer to use. Defaults to torch.nn.Identity.
            activation (torch.nn.Module, optional): The activation function to use. Defaults to torch.nn.ReLU.
            gcn_args (list[Any], optional): Additional arguments for the GCN layer. Defaults to None.
            gcn_kwargs (dict[str, Any], optional): Additional keyword arguments for the GCN layer. Defaults to None.
            norm_args (list[Any], optional): Additional arguments for the normalizer layer. Defaults to None.
            norm_kwargs (dict[str, Any], optional): Additional keyword arguments for the normalizer layer
            activation_args (list[Any], optional): Additional arguments for the activation function. Defaults to None.
            activation_kwargs (dict[str, Any], optional): Additional keyword arguments for the activation function.
        """
        super().__init__()

        # save parameters
        self.dropout_p = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        # initialize layers
        self.dropout = torch.nn.Dropout(p=dropout, inplace=False)

        self.normalizer = normalizer(
            *(norm_args if norm_args is not None else []),
            **(norm_kwargs if norm_kwargs is not None else {}),
        )

        self.activation = activation(
            *(activation_args if activation_args is not None else []),
            **(activation_kwargs if activation_kwargs is not None else {}),
        )

        self.conv = gcn_type(
            in_channels,
            out_channels,
            *(gcn_args if gcn_args is not None else []),
            **(gcn_kwargs if gcn_kwargs is not None else {}),
        )

        if in_channels != out_channels:
            self.projection = torch.nn.Linear(in_channels, out_channels)
        else:
            self.projection = torch.nn.Identity()

    def forward(
        self, x: torch.tensor, edge_index: torch.tensor, **kwargs
    ) -> torch.tensor:
        """Forward pass for the GNNBlock.
        First apply the graph convolution layer, then normalize and apply the activation function.
        Finally, apply a residual connection and dropout.
        Args:
            x (torch.tensor): The input node features.
            edge_index (torch.tensor): The graph connectivity information.
            edge_weight (torch.tensor, optional): The edge weights. Defaults to None.
            kwargs (dict[Any, Any], optional): Additional keyword arguments for the GNN layer. Defaults to None.

        Returns:
            torch.tensor: The output node features.
        """

        x_res = x
        # convolution, then normalize and apply nonlinearity
        x = self.conv(x, edge_index, **kwargs)
        x = self.normalizer(x)
        x = self.activation(x)

        # Residual connection
        x = x + self.projection(x_res)

        # Apply dropout as regularization
        x = self.dropout(x)

        return x

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GNNBlock":
        """Create a GNNBlock from a configuration dictionary.
        When the config does not have 'dropout', it defaults to 0.3.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the parameters for the GNNBlock.

        Returns:
            GNNBlock: An instance of GNNBlock initialized with the provided configuration.
        """
        return cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            dropout=config.get("dropout", 0.3),
            gcn_type=gnn_layers[config["gcn_type"]],
            normalizer=normalizer_layers[config["normalizer"]],
            activation=activation_layers[config["activation"]],
            gcn_args=config.get("gcn_args", []),
            gcn_kwargs=config.get("gcn_kwargs", {}),
            norm_args=config.get("norm_args", []),
            norm_kwargs=config.get("norm_kwargs", {}),
            activation_args=config.get("activation_args", []),
            activation_kwargs=config.get("activation_kwargs", {}),
        )
