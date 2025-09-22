# torch
import torch
import torch_geometric.nn as tgnn
from . import utils

# quality of life
from typing import Any
from pathlib import Path


class GNNBlock(torch.nn.Module):
    """Graph Neural Network Block. Consists of a GNN layer, a normalizer, an activation function,
    and a residual connection. The gnn-layer is applied first, followed by the normalizer and activation function. The result is then projected from the input dimensions to the output dimensions using a linear layer and added to the original input (residual connection). Finally, dropout is applied for regularization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        gnn_layer_type: type[torch.nn.Module] = tgnn.conv.GCNConv,
        normalizer: type[torch.nn.Module] = torch.nn.Identity,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        gnn_layer_args: list[Any] | None = None,
        gnn_layer_kwargs: dict[str, Any] | None = None,
        norm_args: list[Any] | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        activation_args: list[Any] | None = None,
        activation_kwargs: dict[str, Any] | None = None,
        projection_args: list[Any] | None = None,
        projection_kwargs: dict[str, Any] | None = None,
    ):
        """Create a GNNBlock instance.

        Args:
            in_dim (int): The dimensions of the input features.
            out_dim (int): The dimensions of the output features.
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
            projection_args (list[Any], optional): Additional arguments for the projection layer. Defaults to None.
            projection_kwargs (dict[str, Any], optional): Additional keyword arguments for the projection layer. Defaults to None.

        """
        super().__init__()

        # save parameters
        self.dropout_p = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

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
            in_dim,
            out_dim,
            *(gnn_layer_args if gnn_layer_args is not None else []),
            **(gnn_layer_kwargs if gnn_layer_kwargs is not None else {}),
        )

        if in_dim != out_dim:
            self.projection = torch.nn.Linear(
                *(projection_args if projection_args is not None else []),
                **(projection_kwargs if projection_kwargs is not None else {}),
            )
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

        # convolution, then normalize and apply nonlinearity
        x_res = self.conv(x, edge_index, **kwargs)
        x_res = self.normalizer(x_res)
        x_res = self.activation(x_res)

        # Residual connection
        x_res = x_res + self.projection(x)

        # Apply dropout as regularization
        x_res = self.dropout(x_res)

        return x_res

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
            in_dim=config["in_dim"],
            out_dim=config["out_dim"],
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
            projection_args=config.get("projection_args", []),
            projection_kwargs=config.get("projection_kwargs", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save the model's state to file.

        Args:
            path (str | Path): path to save the model to.
        """

        torch.save(self, path)

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device = torch.device("cpu")
    ) -> "GNNBlock":
        """Load a mode instance from file

        Args:
            path (str | Path): Path to the file to load.
            device (torch.device): device to put the model to. Defaults to torch.device("cpu")
        Returns:
            GNNBlock: A GNNBlock instance initialized from the data loaded from the file.
        """

        model = torch.load(path, map_location=device, weights_only=False)
        return model
