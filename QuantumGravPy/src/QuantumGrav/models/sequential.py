import torch
import torch_geometric

from typing import Any, Dict, Tuple
from jsonschema import validate

from .skipconnection import SkipConnection
from .. import base


class Sequential(torch.nn.Module, base.Configurable):
    """Composable sequential model for graph networks.

    Wraps `torch_geometric.nn.Sequential` with configuration support and an optional
    skip connection from the first layer input to the final output. Layers are
    specified as 4-tuples `[module, args, kwargs, signature]`, where `module` is a
    layer class, `args/kwargs` are its constructor parameters, and `signature` is
    the layer's forward signature fragment to be used by PyG's Sequential.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Sequential Model Configuration",
        "type": "object",
        "properties": {
            "layers": {
                "type": "array",
                "description": "Ordered list of layer specs as 4-item arrays: [module, args, kwargs, signature]",
                "items": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": [
                        {
                            "description": "Layer class/module reference (pyobject tag or dotted path string)",
                        },
                        {
                            "type": "array",
                            "description": "Positional arguments for layer constructor",
                            "items": {},
                        },
                        {
                            "type": "object",
                            "description": "Keyword arguments for layer constructor",
                            "additionalProperties": {},
                        },
                        {
                            "type": "string",
                            "description": "Forward signature fragment for this layer (e.g., 'x, edge_index')",
                        },
                    ],
                },
            },
            "forward_signature": {
                "type": "string",
                "description": "Overall forward signature for torch_geometric.nn.Sequential",
                "default": "x, edge_index, batch",
            },
            "with_skip": {
                "type": "boolean",
                "description": "Wether to use a skip connection from first to last layer or not",
                "default": False,
            },
            "skip_args": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "description": "Arguments for the skip connection layer",
                "items": [
                    {
                        "type": "integer",
                        "description": "Size of input features",
                        "minimum": 0,
                    },
                    {
                        "type": "integer",
                        "description": "Size of output features",
                        "minimum": 0,
                    },
                ],
            },
            "skip_kwargs": {
                "type": "object",
                "description": "Optional keyword arguments for the skip connection layer",
                "additionalProperties": {},
            },
        },
        "required": ["layers"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        layers: list[Tuple[type, list[Any], Dict[str, Any], str]],
        forward_signature: str = "x, edge_index, batch",
        with_skip: bool = False,
        skip_args: list[Any] | None = None,
        skip_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the sequential model.

        Args:
            layers (list[Tuple[type, list[Any], Dict[str, Any], str]]):
                Ordered layer specifications as `[module, args, kwargs, signature]`.
                - `module`: Layer class to instantiate.
                - `args`: Positional constructor arguments.
                - `kwargs`: Keyword constructor arguments.
                - `signature`: Forward signature fragment used by PyG Sequential.
            forward_signature (str, optional): Overall forward signature for the
                composed model, passed to `torch_geometric.nn.Sequential`.
                Defaults to "x, edge_index, batch".
            with_skip (bool, optional): If True, add a skip connection from the
                initial input to the final output. Defaults to False.
            skip_args (list[Any] | None, optional): Arguments for the
                `SkipConnection` constructor, expected shape `[in_channels, out_channels]`.
                Required when `with_skip=True`. Defaults to None.
            skip_kwargs (dict[str, Any] | None, optional): Optional keyword args
                for the `SkipConnection` constructor (e.g., `weight_initializer`).
                Defaults to None.
        """
        super().__init__()

        model_layers: list[Tuple[torch.nn.Module, str]] = []

        for layer_specs in layers:
            module, args, kwargs, signature = layer_specs
            model_layers.append(
                (
                    module(
                        *(args if args is not None else []),
                        **(kwargs if kwargs is not None else {}),
                    ),
                    signature,
                )
            )

        self.layers = torch_geometric.nn.Sequential(forward_signature, model_layers)

        if with_skip:
            self.skipconnection = SkipConnection(
                *(skip_args if skip_args is not None else []),
                **(skip_kwargs if skip_kwargs is not None else {}),
            )
        self.with_skip = with_skip

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run the forward pass.

        Applies the composed sequential layers. If a skip connection is
        configured, combines the original input `x` with the sequential output
        via `SkipConnection`.

        Args:
            x (torch.Tensor): Node feature tensor (input to the first layer).
            edge_index (torch.Tensor): Edge indices for the graph.
            *args: Additional positional arguments passed to the sequential.
            **kwargs: Additional keyword arguments passed to the sequential.

        Returns:
            torch.Tensor: The model output, optionally combined with a skip connection.
        """
        if self.with_skip:
            x_ = self.layers.forward(x, edge_index, *args, **kwargs)
            x_ = self.skipconnection(x, x_)
            return x_
        else:
            return self.layers.forward(x, edge_index, *args, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[Any, Any]) -> "Sequential":
        """Construct a `Sequential` instance from a configuration dictionary.

        Validates the `config` against `Sequential.schema` and builds the
        corresponding model, including an optional skip connection.

        Args:
            config (Dict[Any, Any]): Configuration containing keys:
                - `layers`: list of layer specs `[module, args, kwargs, signature]` (required)
                - `forward_signature`: overall forward signature (optional)
                - `with_skip`: enable skip connection (optional)
                - `skip_args`: arguments for `SkipConnection` (optional; required if `with_skip`)
                - `skip_kwargs`: keyword args for `SkipConnection` (optional)

        Returns:
            Sequential: A configured sequential model instance.
        """
        validate(config, cls.schema)

        return cls(
            config["layers"],
            config.get("forward_signature", "x, edge_index, batch"),
            config.get("with_skip", False),
            config.get("skip_args"),
            config.get("skip_kwargs"),
        )
