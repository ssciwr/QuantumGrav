import torch
from torch.nn import Linear


class GraphFeaturesBlock(torch.nn.Module):
    """
    A block for processing graph features in a neural network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.5,
        activation: torch.nn.Module = torch.nn.ReLU(),
        linear_kwargs: dict = None,
        final_linear_kwargs: dict = None,
    ):
        """
        Initializes the GraphFeaturesBlock.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            hidden_dims (list): List of hidden layer dimensions.
            dropout (float): Dropout rate for regularization.
            activation (torch.nn.Module): Activation function to use.
            linear_kwargs (list, optional): List of keyword arguments for each linear layer.
            final_linear_kwargs (dict, optional): Keyword arguments for the final linear layer.
        """

        super(GraphFeaturesBlock, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.hidden_dims = hidden_dims

        if len(hidden_dims) == 0:
            self.linear = Linear(input_dim, output_dim)
        else:
            layers = []
            in_dim = input_dim
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(
                    Linear(
                        in_dim,
                        hidden_dim,
                        **(
                            linear_kwargs[i]
                            if linear_kwargs and linear_kwargs[i]
                            else {}
                        ),
                    )
                )
                layers.append(activation)
                in_dim = hidden_dim
            layers.append(
                Linear(
                    in_dim,
                    output_dim,
                    **(final_linear_kwargs if final_linear_kwargs else {}),
                )
            )
            self.linear = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GraphFeaturesBlock.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.linear(x)
        return x
