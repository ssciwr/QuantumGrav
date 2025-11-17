# torch
import torch
from torch_geometric.nn.sequential import Sequential

# quality of life
from typing import Any, Sequence, Tuple, Dict
from pathlib import Path
import jsonschema

from .base import BaseModel


class SequentialModel(BaseModel):
    """Sequential model block build on top of torch_geometric.nn.sequential.Sequential. Consists of a GNN layer, a normalizer, an activation function,
    and a residual connection. The gnn-layer is applied first, followed by the normalizer and activation function. The result is then projected from the input dimensions to the output dimensions using a linear layer and added to the original input (residual connection). Finally, dropout is applied for regularization.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Sequential model configuration",
        "type": "object",
        "properties": {
            "input_sig": {
                "type": "string",
                "description": "Signature of the model's forward method as required by torch_geometric.nn.sequential.Sequential",
            },
            "layer_specs": {
                "type": "array",
                "items": [
                    {
                        "type": "string",
                        "description": "Signature of the torch.nn.Module layer's forward method",
                    },
                    {
                        "type": "string",
                        "description": "name of the torch.nn.Module layer",
                    },
                    {
                        "type": "array",
                        "description": "Positional arguments",
                        "items": {},
                    },
                    {
                        "type": "object",
                        "description": "Keyword arguments",
                        "additionalProperties": True,
                    },
                ],
                "additionalItems": True,  # allows extra elements beyond the first 3
            },
        },
        "required": [
            "input_sig",
            "layer_specs",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        input_sig: str,
        layer_specs: Sequence[
            Tuple[str, type[torch.nn.Module], Sequence[Any], Dict[str, Any]]
        ],
    ):
        """Build a new SequentialModel which stacks layers sequentially.

        Args:
            input_sig (str): Input signature for the Sequential model. See torch_geometric.nn.sequential.Sequential for details.
            layer_specs (Sequence[ Tuple[str, type[torch.nn.Module], Sequence[Any], dict[str, Any]] ]): Specifications for each layer in the Sequential model.
            This is of the form
            (signature, layer_type, layer_args, layer_kwargs) where:
                - signature (str): Signature for the layer. See torch_geometric.nn.sequential.Sequential for details.
                - layer_type (type[torch.nn.Module]): The class of the layer to instantiate.
                - layer_args (Sequence[Any]): Positional arguments to pass to the layer constructor.
                - layer_kwargs (dict[str, Any]): Keyword arguments to pass to the layer constructor.
        """
        super().__init__()

        self.layerspecs = layer_specs
        self.input_sig = input_sig

        layers = [
            (layer_type(*layer_args, **layer_kwargs), sig)
            if sig
            else (layer_type(*layer_args, **layer_kwargs))
            for (sig, layer_type, layer_args, layer_kwargs) in layer_specs
        ]

        self.layers = Sequential(input_sig, layers)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward pass for the GNNBlock.
        First apply the graph convolution layer, then normalize and apply the activation function.
        Finally, apply a residual connection and dropout.
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The graph connectivity information.
            edge_weight (torch.Tensor, optional): The edge weights. Defaults to None.
            kwargs (dict[Any, Any], optional): Additional keyword arguments for the GNN layer. Defaults to None.

        Returns:
            torch.Tensor: The output node features.
        """

        return self.layers(x, edge_index, **kwargs)

    @classmethod
    def verify_config(cls, config: Dict[str, Any]) -> bool:
        jsonschema.validate(config, schema=cls.schema)
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SequentialModel":
        """Create a SequentialModel from a configuration dictionary.
        When the config does not have 'dropout', it defaults to 0.3.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the parameters for the SequentialModel.

        Returns:
            SequentialModel: An instance of SequentialModel initialized with the provided configuration.
        """
        input_sig = config["input_sig"]
        layer_specs = []
        for key in sorted(config.keys()):
            if key.startswith("layer_"):
                layer_cfg = config[key]
                layer_type = layer_cfg["type"]
                layer_args = layer_cfg.get("args", [])
                layer_kwargs = layer_cfg.get("kwargs", {})
                signature = layer_cfg.get("signature", None)
                layer_specs.append((signature, layer_type, layer_args, layer_kwargs))

        return cls(input_sig=input_sig, layer_specs=layer_specs)

    def to_config(self) -> Dict[str, Any]:
        """Convert the SequentialModel instance to a configuration dictionary."""
        config: Dict[str, Any] = {
            f"layer_{i}": {
                "type": layer_type.__name__,
                "args": layer_args,
                "kwargs": layer_kwargs,
                "signature": sig,
            }
            if sig
            else {
                "type": layer_type.__name__,
                "args": layer_args,
                "kwargs": layer_kwargs,
            }
            for (i, (sig, layer_type, layer_args, layer_kwargs)) in enumerate(
                self.layerspecs
            )
        }
        config["input_sig"] = self.input_sig
        return config

    def save(self, path: str | Path) -> None:
        """Save the model's state to file.

        Args:
            path (str | Path): path to save the model to.
        """

        torch.save({"config": self.to_config(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls, path: str, device: torch.device = torch.device("cpu")
    ) -> "SequentialModel":
        """Load a SequentialModel instance from file

        Args:
            path (str | Path): Path to the file to load.
            device (torch.device): device to put the model to. Defaults to torch.device("cpu")
        Returns:
            GNNBlock: A GNNBlock instance initialized from the data loaded from the file.
        """

        modeldata = torch.load(path, weights_only=False)
        model = cls.from_config(modeldata["config"]).to(device)
        model.load_state_dict(modeldata["state_dict"])
        return model
