import torch


class ClassifierBlock(torch.nn.Module):
    """Classifier Block for single- or multi-class classification.

    Args:
        torch (Module): PyTorch module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        hidden_dims: list[int],
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        linear_kwargs: list[dict] = None,
        output_kwargs: list[dict] = None,
    ):
        """create a classifier block with a backbone and multiple output layers.All are made up of linear layers with an activation function in between.

        Args:
            input_dim (int): input dimension of the classifier block
            output_dims (list[int]): list of output dimensions for each output layer, i.e., each classification task
            hidden_dims (list[int]): list of hidden dimensions for the backbone
            activation (type[torch.nn.Module], optional): activation function to use. Defaults to torch.nn.ReLU.
            linear_kwargs (list[dict], optional): additional arguments for the linear layers. Defaults to None.
            output_kwargs (list[dict], optional): additional arguments for the output layers. Defaults to None.

        Raises:
            ValueError: If hidden_dims is empty or not a list of integers.
            ValueError: If any output_dim is not a positive integer.
        """
        super().__init__()

        self.activation = activation
        self.hidden_dims = hidden_dims

        # build backbone mappings first
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty list of integers")
        else:
            layers = []
            in_dim = input_dim
            for i, hidden_dim in enumerate(hidden_dims):
                layers.append(
                    torch.nn.Linear(
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

            self.backbone = torch.nn.Sequential(*layers)

        # take care of possible multi-objective classification
        self.classifier_layers = []
        for i, output_dim in enumerate(output_dims):
            if output_dim <= 0:
                raise ValueError(
                    f"output_dims[{i}] must be a positive integer, got {output_dim}"
                )

            classifier_layer = torch.nn.Linear(
                hidden_dims[-1],
                output_dim,
                **(output_kwargs[i] if output_kwargs and output_kwargs[i] else {}),
            )
            self.classifier_layers.append(classifier_layer)

    def forward(
        self,
        x: torch.tensor,
        backbone_kwargs: dict = None,
        classifier_layer_kwargs: list[dict] = None,
    ) -> list[torch.Tensor]:
        """Forward pass through the classifier block.

        Args:
            x (torch.tensor): Input tensor.
            backbone_kwargs (dict, optional): Additional arguments for the backbone. Defaults to None.
            classifier_layer_kwargs (list[dict], optional): Additional arguments for each classifier layer. Defaults to None.

        Returns:
            list[torch.Tensor]: List of output tensors from each classifier layer.
        """
        x = self.backbone(x, **(backbone_kwargs if backbone_kwargs is not None else {}))

        logits = [
            classifier_layer(
                x, **(classifier_layer_kwargs[i] if classifier_layer_kwargs else {})
            )
            for i, classifier_layer in enumerate(self.classifier_layers)
        ]

        return logits
