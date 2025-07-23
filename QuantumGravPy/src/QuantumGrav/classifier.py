import torch


class ClassifierBlock(torch.nn.Module):
    """This class implements a neural network block consisting of a backbone
    (a sequence of linear layers with activation functions) and multiple
    output layers for classification tasks. It supports multi-objective
    classification by allowing multiple output layers, each corresponding
    to a different classification task.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        hidden_dims: list[int],
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        backbone_kwargs: list[dict] = None,
        output_kwargs: list[dict] = None,
        activation_kwargs: list[dict] = None,
    ):
        """Create a classifier block with a backbone and multiple output layers. All layers are of type `Linear` with an activation function in between (the backbone) and a set of output layers for each classification task (output)

        Args:
            input_dim (int): input dimension of the classifier block
            output_dims (list[int]): list of output dimensions for each output layer, i.e., each classification task
            hidden_dims (list[int]): list of hidden dimensions for the backbone
            activation (type[torch.nn.Module], optional): activation function to use. Defaults to torch.nn.ReLU.
            backbone_kwargs (list[dict], optional): additional arguments for the backbone layers. Defaults to None.
            output_kwargs (list[dict], optional): additional arguments for the output layers. Defaults to None.

        Raises:
            ValueError: If hidden_dims is empty or not a list of integers.
            ValueError: If any output_dim is not a positive integer.
        """
        super().__init__()

        # build backbone mappings first
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list of integers")

        if not all(h > 0 for h in hidden_dims):
            raise ValueError("hidden_dims must be a list of positive integers")

        # build backbone mappings first
        if len(output_dims) == 0:
            raise ValueError("output_dims must be a non-empty list of integers")

        if not all(o > 0 for o in output_dims):
            raise ValueError("output_dims must be a list of positive integers")

        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(
                torch.nn.Linear(
                    in_dim,
                    hidden_dim,
                    **(
                        backbone_kwargs[i]
                        if backbone_kwargs and backbone_kwargs[i]
                        else {}
                    ),
                )
            )
            layers.append(
                activation(
                    **(
                        activation_kwargs[i]
                        if activation_kwargs and activation_kwargs[i]
                        else {}
                    ),
                )
            )
            in_dim = hidden_dim

        self.backbone = torch.nn.Sequential(*layers)

        # take care of possible multi-objective classification
        output_layers = []
        for i, output_dim in enumerate(output_dims):
            output_layer = torch.nn.Linear(
                hidden_dims[-1],
                output_dim,
                **(output_kwargs[i] if output_kwargs and output_kwargs[i] else {}),
            )
            output_layers.append(output_layer)
        self.output_layers = torch.nn.ModuleList(output_layers)

    def forward(
        self,
        x: torch.tensor,
    ) -> list[torch.Tensor]:
        """Forward pass through the classifier block.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            list[torch.Tensor]: List of output tensors from each classifier layer.
        """
        x = self.backbone(x)

        logits = [output_layer(x) for output_layer in self.output_layers]

        return logits
