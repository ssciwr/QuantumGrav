import torch
import torch_geometric

from typing import Any, Dict, Tuple
from jsonschema import validate

from .. import base


class Sequential(torch.nn.Module, base.Configurable):
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
                            "oneOf": [
                                {
                                    "type": "string",
                                    "description": "Dotted path to module/class",
                                },
                                {
                                    "type": "object",
                                    "description": "Python object spec",
                                    "additionalProperties": True,
                                },
                            ],
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
        },
        "required": ["layers"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        layers: list[Tuple[type, list[Any], Dict[str, Any], str]],
        forward_signature: str = "x, edge_index, batch",
    ):
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

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.layers.forward(x, edge_index, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[Any, Any]) -> "Sequential":
        validate(config, cls.schema)

        return cls(
            config["layers"], config.get("forward_signature", "x, edge_index, batch")
        )
