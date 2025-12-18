import torch
import torch_geometric
from typing import Any


class RedrawProjection:
    """Model to fulfill performer need for redrawing random feature matrix every n steps to avoid bias"""

    def __init__(
        self,
        model: torch.nn.Module,
        redraw_interval: int | None = None,
    ):
        """_summary_

        Args:
            model (torch.nn.Module): _description_
            redraw_interval (int | None, optional): _description_. Defaults to None.
        """
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """_summary_"""
        if not self.model.training or self.redraw_interval is None:
            return

        if self.num_last_redraw >= self.redraw_interval:
            for module in self.model.modules():
                if isinstance(module, torch_geometric.nn.attention.PerformerAttention):
                    module.redraw_projection_matrix()

            self.num_last_redraw = 0
        else:
            self.num_last_redraw += 1


class GPSModel(torch.nn.Module):
    """GPS Example graph transformer from pyg examples, adjusted to work with QuantumGrav
    In particular, this implementation does not use edge features, and does not
    use positional encodings because they are already included in the node features.
    See [the graph transformer documentation of torch_geometric for more](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        channels: int,
        num_heads: int,
        num_layers: int,
        attn_type: str,
        attn_kwargs: dict[str, Any],
        redraw_interval: int | None = 1000,
    ):
        """Construct a new GPS Model instance

        Args:
            in_features (int): input feature dimensionality
            out_features (int): output feature dimensionality
            channels (int): Input- and output feature dimensionality of the internal Linear layers.
            num_heads (int): number of attention heads
            num_layers (int): number of GPS layers in the model
            attn_type (str): type of attention mechanism to use
            attn_kwargs (dict[str, Any]): additional keyword arguments for the attention mechanism
            redraw_interval (int | None, optional): interval for redrawing random feature matrix in performer attention. Defaults to 1000.
        """
        super().__init__()

        # embeddings
        self.input_proj = torch.nn.Linear(
            in_features=in_features, out_features=channels
        )

        # convolutional part
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            nn = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
            )

            # this is the main part of the architecture
            conv = torch_geometric.nn.GPSConv(
                channels,
                torch_geometric.nn.GINConv(
                    nn,
                    train_eps=True,
                ),
                heads=num_heads,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs,
            )

            self.convs.append(conv)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // 2, channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // 4, out_features),
        )

        # README: what do we need this for?
        self.redraw = RedrawProjection(
            self.convs,
            redraw_interval=redraw_interval if attn_type == "performer" else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GPS Model forward pass

        Args:
            x (torch.Tensor): input node features
            edge_index (torch.Tensor): edge indices
            batch (torch.Tensor | None, optional): batch vector. Defaults to None.

        Returns:
            torch.Tensor: output node features
        """
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)

        x = torch_geometric.nn.global_add_pool(x, batch)

        x = self.mlp(x)
        return x


class GPSTransformer(torch.nn.Module):
    """GPS Transformer wrapper for QuantumGravPy"""

    def __init__(
        self,
        # transformer model args
        in_features: int,
        out_features: int,
        channels: int,
        num_heads: int,
        num_layers: int,
        attn_type: str,
        attn_kwargs: dict[str, Any],
        # redraw attention random features matrix args
        redraw_interval: int | None = None,
    ):
        """Instantiate a new GPSTransformer model

        Args:
            in_features (int): input feature dimensionality
            out_features (int): output feature dimensionality
            channels (int): Input- and output feature dimensionality of the internal Linear layers.
            num_heads (int): number of attention heads
            num_layers (int): number of GPS layers in the model
            attn_type (str): type of attention mechanism to use
            attn_kwargs (dict[str, Any]): additional keyword arguments for the attention mechanism
            redraw_interval (int | None, optional): interval for redrawing random feature matrix in performer attention. Defaults to 1000.
        """
        super().__init__()

        self.transformer_model = GPSModel(
            in_features,
            out_features,
            channels,
            num_heads,
            num_layers,
            attn_type,
            attn_kwargs,
            redraw_interval=redraw_interval,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ):
        """GPS Transformer forward pass

        Args:
            x (torch.Tensor): input node features
            edge_index (torch.Tensor): edge indices
            batch (torch.Tensor | None, optional): batch vector. Defaults to None.

        Returns:
            torch.Tensor: output node features
        """
        self.transformer_model.redraw.redraw_projections()
        return self.transformer_model(x, edge_index, batch)
