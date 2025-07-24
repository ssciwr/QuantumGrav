# torch
import torch
import torch_geometric.nn as tgnn
from . import utils
# quality of life
from typing import Any


class GNNBlock(torch.nn.Module):
    """Graph Neural Network Block. Consists of a GNN layer, a normalizer, an activation function,
    and a residual connection. The gnn-layer is applied first, followed by the normalizer and activation function. The result is then projected from the input dimensions to the output dimensions using a linear layer and added to the original input (residual connection). Finally, dropout is applied for regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        gnn_layer_type: torch.nn.Module = tgnn.conv.GCNConv,
        normalizer: torch.nn.Module = torch.nn.Identity,
        activation: torch.nn.Module = torch.nn.ReLU,
        gnn_layer_args: list[Any] = None,
        gnn_layer_kwargs: dict[str, Any] = None,
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
            gnn_layer_type (torch.nn.Module, optional): The type of GNN-layer to use. Defaults to tgnn.conv.GCNConv.
            normalizer (torch.nn.Module, optional): The normalizer layer to use. Defaults to torch.nn.Identity.
            activation (torch.nn.Module, optional): The activation function to use. Defaults to torch.nn.ReLU.
            gnn_layer_args (list[Any], optional): Additional arguments for the GNN layer. Defaults to None.
            gnn_layer_kwargs (dict[str, Any], optional): Additional keyword arguments for the GNN layer. Defaults to None.
            norm_args (list[Any], optional): Additional arguments for the normalizer layer. Defaults to None.
            norm_kwargs (dict[str, Any], optional): Additional keyword arguments for the normalizer layer. Defaults to None.
            activation_args (list[Any], optional): Additional arguments for the activation function. Defaults to None.
            activation_kwargs (dict[str, Any], optional): Additional keyword arguments for the activation function. Defaults to None.
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

        self.conv = gnn_layer_type(
            in_channels,
            out_channels,
            *(gnn_layer_args if gnn_layer_args is not None else []),
            **(gnn_layer_kwargs if gnn_layer_kwargs is not None else {}),
        )

        if in_channels != out_channels:
            self.projection = torch.nn.Linear(in_channels, out_channels)
        else:
            self.projection = torch.nn.Identity()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward pass for the GNNBlock.
        First apply the graph convolution layer, then normalize and apply the activation function.
        Finally, apply a residual connection and dropout.
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The graph connectivity information.
            edge_weight (torch.Tensor, optional): The edge weights. Defaults to None.
            kwargs (dict[Any, Any], optional): Additional keyword arguments for the GNN layer. Defaults to None.

        Returns:
            torch.Tensor: The output node features.
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
            gnn_layer_type=utils.gnn_layers[config["gnn_layer_type"]],
            normalizer=utils.normalizer_layers[config["normalizer"]],
            activation=utils.activation_layers[config["activation"]],
            gnn_layer_args=config.get("gnn_layer_args", []),
            gnn_layer_kwargs=config.get("gnn_layer_kwargs", {}),
            norm_args=config.get("norm_args", []),
            norm_kwargs=config.get("norm_kwargs", {}),
            activation_args=config.get("activation_args", []),
            activation_kwargs=config.get("activation_kwargs", {}),
        )
