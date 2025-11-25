from .. import base
import torch
import torch_geometric

from jsonschema import validate

from typing import Any, Dict


class SkipConnection(torch.nn.Module, base.Configurable):
    """A skip connection wrapped in a torch.nn.Module."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SkipConnection Configuration",
        "type": "object",
        "properties": {
            "in_channels": {
                "type": "integer",
                "description": "Size of input features",
                "minimum": 0,
            },
            "out_channels": {
                "type": "integer",
                "description": "Size of output features",
                "minimum": 0,
            },
            "weight_initializer": {
                "type": "string",
                "description": "Initializer for the weight matrix",
                "enum": ["glorot", "uniform", "kaiming_uniform"],
            },
            "bias_initializer": {
                "type": "string",
                "description": "Initializer for the bias vector",
                "enum": ["zeros"],
            },
        },
        "required": ["in_channels", "out_channels"],
        "additionalProperties": False,
    }

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

        self.cfg = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
        }

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

    @classmethod
    def verify_config(cls, config: Dict[str, Any]) -> bool:
        """_summary_

        Args:
            config (Dict[str, Any]): _description_

        Returns:
            bool: _description_
        """
        validate(config, cls.schema)

        # Returns True only if validate does not raise a ValidationError
        return True

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SkipConnection":
        """Construct a SkipConnection instance from config

        Args:
            config (Dict[str, Any]): config to construct a new instance from

        Raises:
            RuntimeError: When the config is not valid, i.e., contains the wrong types or does not the provide all needed elements

        Returns:
            SkipConnection: new SkipConnection instance.
        """
        try:
            cls.verify_config(config)

            return cls(
                config["in_channels"],
                config["out_channels"],
                weight_initializer=config["weight_initializer"],
                bias_initializer=config["bias_initializer"],
            )
        except Exception as e:
            raise RuntimeError(f"Error, couldn't build SkipConnection: {e}")

    def to_config(self) -> Dict[str, Any]:
        """Return a config representation of the caller

        Returns:
            Dict[str, Any]: Dictionary containing everything the caller has been constructed from.
        """
        return self.cfg
