import torch
import torch_geometric

from . import utils
from typing import Any


class LinearSequential(torch.nn.Module):
    """This class implements a neural network block consisting of a backbone
    (a sequence of linear layers with activation functions) and multiple
    output layers for classification tasks. It supports multi-objective
    classification by allowing multiple output layers, each corresponding
    to a different classification task, but can also be used for any other type of sequential processing that involves linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        hidden_dims: list[int] | None = None,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        backbone_kwargs: list[dict] | None = None,
        output_kwargs: list[dict] | None = None,
        activation_kwargs: list[dict] | None = None,
    ):
        """Create a LinearSequential object with a backbone and multiple output layers. All layers are of type `Linear` with an activation function in between (the backbone) and a set of linear output layers.

        Args:
            input_dim (int): input dimension of the LinearSequential object
            output_dims (list[int]): list of output dimensions for each output layer, i.e., each classification task
            hidden_dims (list[int]): list of hidden dimensions for the backbone
            activation (type[torch.nn.Module], optional): activation function to use. Defaults to torch.nn.ReLU.
            backbone_kwargs (list[dict], optional): additional arguments for the backbone layers. Defaults to None.
            output_kwargs (list[dict], optional): additional arguments for the output layers. Defaults to None.

        Raises:
            ValueError: If hidden_dims contains non-positive integers.
            ValueError: If output_dims is empty or contains non-positive integers.
            ValueError: If any output_dim is not a positive integer.
        """
        super().__init__()

        # validate input parameters
        if hidden_dims is None:
            raise ValueError("hidden_dims must not be None")

        if not all(h > 0 for h in hidden_dims):
            raise ValueError("hidden_dims must be a list of positive integers")

        if len(output_dims) == 0:
            raise ValueError("output_dims must be a non-empty list of integers")

        if not all(o > 0 for o in output_dims):
            raise ValueError("output_dims must be a list of positive integers")

        # manage kwargs for the different parts of the network
        processed_backbone_kwargs = self._handle_kwargs(
            backbone_kwargs, "backbone_kwargs", len(hidden_dims)
        )

        processed_activation_kwargs = self._handle_kwargs(
            activation_kwargs, "activation_kwargs", len(hidden_dims)
        )

        processed_output_kwargs = self._handle_kwargs(
            output_kwargs, "output_kwargs", len(output_dims)
        )

        # build backbone with Sequential
        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(
                torch_geometric.nn.dense.Linear(
                    in_dim,
                    hidden_dim,
                    **processed_backbone_kwargs[i],
                )
            )
            layers.append(
                activation(
                    **processed_activation_kwargs[i],
                )
            )
            in_dim = hidden_dim

        if len(layers) > 0:
            self.backbone = torch.nn.Sequential(*layers)
        else:
            self.backbone = torch.nn.Identity()

        # build the final layers - take care of possible multi-objective classification
        output_layers = []

        final_in_dim = (
            hidden_dims[-1] if hidden_dims and len(hidden_dims) > 0 else input_dim
        )

        for i, output_dim in enumerate(output_dims):
            output_layer = torch_geometric.nn.dense.Linear(
                final_in_dim,
                output_dim,
                **(
                    processed_output_kwargs[i]
                    if processed_output_kwargs and processed_output_kwargs[i]
                    else {}
                ),
            )
            output_layers.append(output_layer)
        self.output_layers = torch.nn.ModuleList(output_layers)

    def _handle_kwargs(
        self, kwarglist: list[dict] | None, name: str, needed: int
    ) -> list[dict]:
        """
        handle kwargs for the backbone or activation functions.
        """
        if kwarglist is None:
            kwarglist = [{}] * needed
        elif len(kwarglist) == 1:
            kwarglist = kwarglist * needed
        elif len(kwarglist) != needed:
            raise ValueError(
                f"{name} must be a list of dictionaries with the same length as hidden_dims"
            )
        return kwarglist

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass through the LinearSequential object.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            list[torch.Tensor]: List of output tensors from each classifier layer.
        """
        # Sequential handles passing output from one layer to the next
        features = self.backbone(x)  # No need for manual looping or cloning

        # Apply each output layer to the backbone output
        logits = [layer(features) for layer in self.output_layers]

        return logits

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LinearSequential":
        """Create a LinearSequential from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary containing parameters for the LinearSequential.

        Returns:
            LinearSequential: An instance of LinearSequential initialized with the provided configuration.
        """
        return cls(
            input_dim=config["input_dim"],
            output_dims=config["output_dims"],
            hidden_dims=config["hidden_dims"],
            activation=utils.get_registered_activation(
                config.get("activation", "relu")
            ),
            backbone_kwargs=config.get("backbone_kwargs", None),
            output_kwargs=config.get("output_kwargs", None),
            activation_kwargs=config.get("activation_kwargs", None),
        )
