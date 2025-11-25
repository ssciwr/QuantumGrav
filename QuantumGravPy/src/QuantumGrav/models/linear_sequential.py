import torch
import torch_geometric

import logging
from typing import Any, Sequence
from pathlib import Path

from .. import utils


class LinearSequential(torch.nn.Module):
    """This class implements a neural network block consisting of a backbone
    (a sequence of linear layers with activation functions) and multiple
    output layers for classification tasks. It supports multi-objective
    classification by allowing multiple output layers, each corresponding
    to a different classification task, but can also be used for any other type of sequential processing that involves linear layers.
    """

    def __init__(
        self,
        dims: list[Sequence[int]],
        activations: list[type[torch.nn.Module]] = [torch.nn.ReLU],
        linear_kwargs: list[dict] | None = None,
        activation_kwargs: list[dict] | None = None,
    ):
        """Create a LinearSequential object with a backbone and multiple output layers. All layers are of type `Linear` with an activation function in between (the backbone) and a set of linear output layers.

        Args:
            input_dim (int): input dimension of the LinearSequential object
            output_dim (int): output dimension for the output layer, i.e., the classification task
            hidden_dims (list[int]): list of hidden dimensions for the backbone
            activation (type[torch.nn.Module], optional): activation function to use. Defaults to torch.nn.ReLU.
            backbone_kwargs (list[dict], optional): additional arguments for the backbone layers. Defaults to None.
            output_kwargs (dict, optional): additional keyword arguments for the output layers. Defaults to None.

        Raises:
            ValueError: If hidden_dims contains non-positive integers.
            ValueError: If output_dim is a non-positive integer.
        """
        super().__init__()

        if len(dims) == 0:
            raise ValueError("dims must not be empty")

        if len(dims) != len(activations):
            raise ValueError("dims and activations must have the same length")

        if linear_kwargs is None:
            linear_kwargs = [{} for _ in range(len(dims))]

        if activation_kwargs is None:
            activation_kwargs = [{} for _ in range(len(dims))]

        if len(linear_kwargs) != len(dims):
            raise ValueError("linear_kwargs must have the same length as dims")

        if len(activation_kwargs) != len(dims):
            raise ValueError("activation_kwargs must have the same length as dims")

        # build backbone with Sequential
        layers = []
        for i in range(len(dims)):
            in_dim = dims[i][0]
            out_dim = dims[i][1]
            layers.append(
                torch_geometric.nn.dense.Linear(
                    in_dim,
                    out_dim,
                    **linear_kwargs[i],
                )
            )
            layers.append(
                activations[i](
                    **activation_kwargs[i],
                )
            )

        self.layers = torch.nn.Sequential(*layers)
        self.linear_kwargs = linear_kwargs
        self.activation_kwargs = activation_kwargs
        self.logger = logging.getLogger(__name__)

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
        logits = self.layers(x)  # No need for manual looping or cloning

        return logits

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LinearSequential":
        """Create a LinearSequential from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary containing parameters for the LinearSequential.

        Returns:
            LinearSequential: An instance of LinearSequential initialized with the provided configuration.

        Raises:
            ValueError: If the specified activation function is not registered.
        """
        activations = config["activations"]
        activations = [utils.get_registered_activation(act) for act in activations]

        if None in activations:
            raise ValueError(
                f"Activation function '{config.get('activation')}' is not registered."
            )

        return cls(
            dims=config["dims"],
            activations=activations,
            linear_kwargs=config.get("linear_kwargs", None),
            activation_kwargs=config.get("activation_kwargs", None),
        )

    def to_config(self) -> dict[str, Any]:
        """Build a config file from the current model

        Returns:
            dict[str, Any]: Model config
        """
        linear_dims = []
        activations = []

        for layer in self.layers:
            if isinstance(layer, torch_geometric.nn.dense.Linear):
                linear_dims.append((layer.in_channels, layer.out_channels))
            elif isinstance(layer, torch.nn.Linear):
                linear_dims.append((layer.in_features, layer.out_features))
            elif isinstance(layer, torch.nn.Module):
                activations.append(utils.activation_layers_names[type(layer)])
            else:
                self.logger.warning(f"Unknown layer type: {type(layer)}")

        config = {
            "dims": linear_dims,
            "activations": activations,
            "linear_kwargs": self.linear_kwargs,
            "activation_kwargs": self.activation_kwargs,
        }

        return config

    def save(self, path: str | Path) -> None:
        """Save the model's state to file.

        Args:
            path (str | Path): path to save the model to.
        """

        self_as_config = self.to_config()

        torch.save({"config": self_as_config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device = torch.device("cpu")
    ) -> "LinearSequential":
        """Load a LinearSequential instance from file

        Args:
            path (str | Path): path to the file to load the model from
            device (torch.device): device to put the model to. Defaults to torch.device("cpu")
        Returns:
            LinearSequential: An instance of LinearSequential initialized from the loaded data.
        """
        cfg = torch.load(path)

        model = cls.from_config(cfg["config"])
        model.load_state_dict(cfg["state_dict"], strict=False)
        model.to(device)

        return model
