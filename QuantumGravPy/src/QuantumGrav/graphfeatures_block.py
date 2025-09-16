import torch
from . import utils
from . import linear_sequential as QGLS


class GraphFeaturesBlock(QGLS.LinearSequential):
    """Graph Features Block for processing global graph features. Similarly to the classifier, this consists of a sequence of linear layers with  activation functions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        layer_kwargs: list[dict] | None = None,
        activation_kwargs: dict | None = None,
    ):
        """Create a GraphFeaturesBlock instance. This will create at least one hidden layer and one output layer, with the specified input and output dimensions.

        Args:
            input_dim (int): input dimension of the GraphFeaturesBlock
            output_dim (int): output dimension of the GraphFeaturesBlock
            hidden_dims (list[int], optional): output dimensions of the hidden layers. Defaults to None.
            activation (torch.nn.Module, optional): activation function type, e.g., torch.nn.ReLU. Defaults to torch.nn.ReLU.
            layer_kwargs (list[dict], optional): keyword arguments for the constructors of each layer. Defaults to None.
            activation_kwargs (dict, optional): keyword arguments for the construction of each activation function. Defaults to None.
        """
        super().__init__(
            input_dim=input_dim,
            output_dims=[
                output_dim,
            ],
            hidden_dims=hidden_dims,
            activation=activation,
            backbone_kwargs=layer_kwargs,
            output_kwargs=[
                layer_kwargs[-1] if layer_kwargs and len(layer_kwargs) > 0 else None
            ],
            activation_kwargs=activation_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GraphFeaturesBlock.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        res = super().forward(x)
        return res[0] if isinstance(res, list) else res

    @classmethod
    def from_config(cls, config: dict) -> "GraphFeaturesBlock":
        """Create a GraphFeaturesBlock from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the block.

        Returns:
            GraphFeaturesBlock: An instance of GraphFeaturesBlock.
        """
        return cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dims=config.get("hidden_dims", []),
            activation=utils.activation_layers[config["activation"]],
            layer_kwargs=config.get("layer_kwargs", []),
            activation_kwargs=config.get("activation_kwargs", {}),
        )
