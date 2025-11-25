# torch
import torch
import torch_geometric
from . import skipconnection
from .. import base

# quality of life
from typing import Any
from pathlib import Path


class GNNBlock(torch.nn.Module, base.Configurable):
    """Graph Neural Network Block. Consists of a GNN layer, a normalizer, an activation function,
    and a residual connection. The gnn-layer is applied first, followed by the normalizer and activation function. The result is then projected from the input dimensions to the output dimensions using a linear layer and added to the original input (residual connection). Finally, dropout is applied for regularization.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GNNBlock Configuration",
        "type": "object",
        "properties": {},
        "required": ["tasks", "mode", "patience"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        with_skip: bool = True,
        gnn_layer_type: type[torch.nn.Module] = torch_geometric.nn.conv.GCNConv,
        normalizer_type: type[torch.nn.Module] = torch.nn.Identity,
        activation_type: type[torch.nn.Module] = torch.nn.ReLU,
        gnn_layer_args: list[Any] | None = None,
        gnn_layer_kwargs: dict[str, Any] | None = None,
        norm_args: list[Any] | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        activation_args: list[Any] | None = None,
        activation_kwargs: dict[str, Any] | None = None,
        skip_args: list[Any] | None = None,
        skip_kwargs: dict[str, Any] | None = None,
    ):
        """Create a GNNBlock instance.

        Args:
            in_dim (int): The dimensions of the input features.
            out_dim (int): The dimensions of the output features.
            dropout (float, optional): The dropout probability. Defaults to 0.3.
            with_skip (bool, optional): Whether to use a skip connection. Defaults to True.
            gnn_layer_type (torch.nn.Module, optional): The type of GNN-layer to use. Defaults to torch_geometric.nn.conv.GCNConv.
            normalizer (torch.nn.Module, optional): The normalizer layer to use. Defaults to torch.nn.Identity.
            activation (torch.nn.Module, optional): The activation function to use. Defaults to torch.nn.ReLU.
            gnn_layer_args (list[Any], optional): Additional arguments for the GNN layer. Defaults to None.
            gnn_layer_kwargs (dict[str, Any], optional): Additional keyword arguments for the GNN layer. Defaults to None.
            norm_args (list[Any], optional): Additional arguments for the normalizer layer. Defaults to None.
            norm_kwargs (dict[str, Any], optional): Additional keyword arguments for the normalizer layer. Defaults to None.
            activation_args (list[Any], optional): Additional arguments for the activation function. Defaults to None.
            activation_kwargs (dict[str, Any], optional): Additional keyword arguments for the activation function. Defaults to None.
            skip_args (list[Any], optional): Additional arguments for the projection layer. Defaults to None.
            skip_kwargs (dict[str, Any], optional): Additional keyword arguments for the projection layer. Defaults to None.

        """
        super().__init__()

        # save parameters
        self.dropout_p = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.with_skip = with_skip
        # save args/kwargs
        self.gnn_layer_args = gnn_layer_args
        self.gnn_layer_kwargs = gnn_layer_kwargs
        self.norm_args = norm_args
        self.norm_kwargs = norm_kwargs
        self.activation_args = activation_args
        self.activation_kwargs = activation_kwargs
        self.skip_args = skip_args
        self.skip_kwargs = skip_kwargs

        # initialize layers
        self.dropout = torch.nn.Dropout(p=dropout, inplace=False)

        self.normalizer = normalizer_type(
            *(norm_args if norm_args is not None else []),
            **(norm_kwargs if norm_kwargs is not None else {}),
        )

        self.activation = activation_type(
            *(activation_args if activation_args is not None else []),
            **(activation_kwargs if activation_kwargs is not None else {}),
        )

        self.conv = gnn_layer_type(
            in_dim,
            out_dim,
            *(gnn_layer_args if gnn_layer_args is not None else []),
            **(gnn_layer_kwargs if gnn_layer_kwargs is not None else {}),
        )

        if self.skip_kwargs is None:
            self.skip_kwargs = {"bias": False}

        if self.skip_args is None:
            self.skip_args = [in_dim, out_dim]

        if with_skip:
            self.skip = skipconnection.SkipConnection(
                *self.skip_args, *self.skip_kwargs
            )

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
        if self.with_skip:
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
            with_skip=config.get("with_skip", True),
            gnn_layer_type=config["gnn_layer_type"],
            normalizer=config["normalizer_type"],
            activation=config["activation_type"],
            gnn_layer_args=config.get("gnn_layer_args", []),
            gnn_layer_kwargs=config.get("gnn_layer_kwargs", {}),
            norm_args=config.get("norm_args", []),
            norm_kwargs=config.get("norm_kwargs", {}),
            activation_args=config.get("activation_args", []),
            activation_kwargs=config.get("activation_kwargs", {}),
            skip_args=config.get("skip_args", None),
            skip_kwargs=config.get("skip_kwargs", None),
        )

    def to_config(self) -> dict[str, Any]:
        """Convert the GNNBlock instance to a configuration dictionary."""
        config = {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "dropout": self.dropout.p,
            "with_skip": self.with_skip,
            "gnn_layer_type": type(self.conv).__name__,
            "gnn_layer_args": self.gnn_layer_args,
            "gnn_layer_kwargs": self.gnn_layer_kwargs,
            "normalizer_type": type(self.normalizer).__name__,
            "norm_args": self.norm_args,
            "norm_kwargs": self.norm_kwargs,
            "activation_type": type(self.activation).__name__,
            "activation_args": self.activation_args,
            "activation_kwargs": self.activation_kwargs,
            "skip_args": self.skip_args,
            "skip_kwargs": self.skip_kwargs,
        }
        return config

    def save(self, path: str | Path) -> None:
        """Save the model's state to file.

        Args:
            path (str | Path): path to save the model to.
        """

        self_as_cfg = self.to_config()

        torch.save({"config": self_as_cfg, "state_dict": self.state_dict()}, path)

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

        modeldata = torch.load(path, weights_only=False)
        model = cls.from_config(modeldata["config"]).to(device)
        model.load_state_dict(modeldata["state_dict"])
        return model
