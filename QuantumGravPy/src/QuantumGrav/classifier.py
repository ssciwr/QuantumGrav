import torch
import torch_geometric

from . import utils
from typing import Any

from . import linear_sequential as QGLS


class ClassifierBlock(QGLS.LinearSequential):
    """This class implements a neural network block consisting of a backbone
    (a sequence of linear layers with activation functions) and multiple
    output layers for classification tasks. It supports multi-objective
    classification by allowing multiple output layers, each corresponding
    to a different classification task.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        hidden_dims: list[int] = None,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        backbone_kwargs: list[dict] = None,
        output_kwargs: list[dict] = None,
        activation_kwargs: list[dict] = None,
    ):
        """Instantiate a ClassifierBlock.

        Args:
            input_dim (int): _description_
            output_dims (list[int]): _description_
            hidden_dims (list[int], optional): _description_. Defaults to None.
            activation (type[torch.nn.Module], optional): _description_. Defaults to torch.nn.ReLU.
            backbone_kwargs (list[dict], optional): _description_. Defaults to None.
            output_kwargs (list[dict], optional): _description_. Defaults to None.
            activation_kwargs (list[dict], optional): _description_. Defaults to None.
        """
        super().__init__(
            input_dim=input_dim,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            activation=activation,
            backbone_kwargs=backbone_kwargs,
            output_kwargs=output_kwargs,
            activation_kwargs=activation_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "ClassifierBlock":
        """Create a ClassifierBlock from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the block.

        Returns:
            ClassifierBlock: An instance of ClassifierBlock.
        """
        return cls(
            input_dim=config["input_dim"],
            output_dims=config["output_dims"],
            hidden_dims=config.get("hidden_dims", []),
            activation=utils.activation_layers[config["activation"]],
            backbone_kwargs=config.get("backbone_kwargs", []),
            output_kwargs=config.get("output_kwargs", []),
            activation_kwargs=config.get("activation_kwargs", {}),
        )
