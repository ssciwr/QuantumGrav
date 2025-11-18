import torch
import torch_geometric


class SkipConnection(torch.nn.Module):
    """A skip connection wrapped in a torch.nn.Module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight_initializer: str | None = None,
        bias_initializer: str | None = None,
    ):
        """Initialize the SkipConnection module. For mismatched input/output dimensions, a linear projection is applied.
        For weight and bias initializers, see torch_geometric.nn.dense.Linear documentation.

        Args:
            in_channels (int): input feature dimension
            out_channels (int): output feature dimension
            weight_initializer (str | None, optional): weight initializer. Defaults to None.
            bias_initializer (str | None, optional): bias initializer. Defaults to None.
        """
        super().__init__()

        if in_channels != out_channels:
            self.proj = torch_geometric.nn.dense.Linear(
                in_channels,
                out_channels,
                bias=False,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        else:
            self.proj = torch.nn.Identity()

    def forward(self, x_old: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SkipConnection module.

        Args:
            x_old (torch.Tensor): input tensor before the skip connection
            x (torch.Tensor): input tensor after the skip connection
        Returns:
            torch.Tensor: output tensor after applying the skip connection
        """
        return x + self.proj(x_old)
