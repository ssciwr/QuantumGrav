import torch
from torch.nn import Linear
import torch.nn.functional as F


class ClassifierBlock(torch.nn.Module):
    """
    A classifier block that can be used in a neural network for classification tasks.
    This block consists of MLP backbone and three classification layers which are MLPs, too:
    - Manifold classification layer
    - Boundary classification layer
    - Dimension classification layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        manifold_classes: int = 6,
        boundary_classes: int = 3,
        dimension_classes: int = 3,
        activation: callable = F.relu,
        linear_kwargs: list[dict] = None,
        dim_kwargs: dict = None,
        boundary_kwargs: dict = None,
        manifold_kwargs: dict = None,
    ):
        """
        Initializes the ClassifierBlock.
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            hidden_dims (list[int]): List of hidden layer dimensions.
            manifold_classes (int): Number of classes for manifold classification.
            boundary_classes (int): Number of classes for boundary classification.
            dimension_classes (int): Number of classes for dimension classification.
            activation (callable): Activation function to apply after each linear layer.
            linear_kwargs (list[dict], optional): Additional keyword arguments for each linear layer.
            dim_kwargs (dict, optional): Additional keyword arguments for the dimension classification layer.
            boundary_kwargs (dict, optional): Additional keyword arguments for the boundary classification layer.
            manifold_kwargs (dict, optional): Additional keyword arguments for the manifold classification layer.
        """

        super(ClassifierBlock, self).__init__()
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.total_classes = manifold_classes + boundary_classes + dimension_classes
        self.manifold_classes = manifold_classes
        self.boundary_classes = boundary_classes
        self.dimension_classes = dimension_classes

        if len(hidden_dims) == 0:
            self.backbone = Linear(
                input_dim, output_dim, **(linear_kwargs[0] if linear_kwargs else {})
            )
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

            self.backbone = torch.nn.Sequential(*layers)

            self.dim_layer = torch.nn.Linear(
                hidden_dim, self.dimension_classes, **(dim_kwargs if dim_kwargs else {})
            )

            self.boundary_layer = torch.nn.Linear(
                hidden_dim,
                self.boundary_classes,
                **(boundary_kwargs if boundary_kwargs else {}),
            )

            self.manifold_layer = torch.nn.Linear(
                hidden_dim,
                self.manifold_classes,
                **(manifold_kwargs if manifold_kwargs else {}),
            )

    def forward(
        self,
        x: torch.Tensor,
        backbone_kwargs: dict = None,
        dim_layer_kwargs: dict = None,
        boundary_layer_kwargs: dict = None,
        manifold_layer_kwargs: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the classifier block.

        Args:
            x (torch.Tensor): Input tensor.
            backbone_kwargs (dict, optional): Keyword arguments for the backbone. Defaults to None.
            dim_layer_kwargs (dict, optional): Keyword arguments for the dimension layer. Defaults to None.
            boundary_layer_kwargs (dict, optional): Keyword arguments for the boundary layer. Defaults to None.
            manifold_layer_kwargs (dict, optional): Keyword arguments for the manifold layer. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors for dimension, boundary, and manifold classifications.
        """

        x = self.backbone(x, **(backbone_kwargs if backbone_kwargs is not None else {}))

        dim_logit = self.dim_layer(
            x, **(dim_layer_kwargs if dim_layer_kwargs is not None else {})
        )
        boundary_logit = self.boundary_layer(
            x, **(boundary_layer_kwargs if boundary_layer_kwargs is not None else {})
        )
        manifold_logit = self.manifold_layer(
            x, **(manifold_layer_kwargs if manifold_layer_kwargs is not None else {})
        )

        return dim_logit, boundary_logit, manifold_logit
