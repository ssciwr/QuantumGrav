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
            "input_signature": {
                "type": "string",
                "description": "Signature of the model's forward method as required by torch_geometric.nn.sequential.Sequential",
            },
            "layer_specs": {
                "type": "array",
                "description": "Array of layer specifications, each containing (signature, layer_name, args, kwargs)",
                "items": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": [
                        {
                            "oneOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Signature of the torch.nn.Module layer's forward method (can be null)",
                        },
                        {
                            "description": "name of the torch.nn.Module layer or class object",
                        },
                        {
                            "type": "array",
                            "description": "Positional arguments",
                        },
                        {
                            "type": "object",
                            "description": "Keyword arguments",
                            "additionalProperties": True,
                        },
                    ],
                },
            },
        },
        "required": [
            "input_signature",
            "layer_specs",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        input_signature: str,
        layer_specs: Sequence[
            Tuple[str, type[torch.nn.Module], Sequence[Any], Dict[str, Any]]
        ],
    ):
        """Build a new SequentialModel which stacks layers sequentially.

        Args:
            input_signature (str): Input signature for the Sequential model. See torch_geometric.nn.sequential.Sequential for details.
            layer_specs (Sequence[ Tuple[str, type[torch.nn.Module], Sequence[Any], dict[str, Any]] ]): Specifications for each layer in the Sequential model.
            This is of the form
            (signature, layer_type, layer_args, layer_kwargs) where:
                - signature (str): Signature for the layer. See torch_geometric.nn.sequential.Sequential for details.
                - layer_type (type[torch.nn.Module]): The class of the layer to instantiate.
                - layer_args (Sequence[Any]): Positional arguments to pass to the layer constructor.
                - layer_kwargs (dict[str, Any]): Keyword arguments to pass to the layer constructor.
        """
        super().__init__()

        self.layerspecs = [list(spec) for spec in layer_specs]
        self.input_signature = input_signature

        layers = [
            (layer_type(*layer_args, **layer_kwargs), sig)
            if sig
            else (layer_type(*layer_args, **layer_kwargs))
            for (sig, layer_type, layer_args, layer_kwargs) in layer_specs
        ]

        self.layers = Sequential(input_signature, layers)

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

        The config should contain:
        - 'input_signature': str - the input signature for the Sequential model
        - 'layer_specs': list of [signature, layer_type, args, kwargs] tuples
          where signature can be str or None, layer_type is a torch.nn.Module class,
          args is a list, and kwargs is a dict.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the parameters for the SequentialModel.

        Returns:
            SequentialModel: An instance of SequentialModel initialized with the provided configuration.
        """
        cls.verify_config(config)

        input_signature = config["input_signature"]
        layer_specs = config["layer_specs"]

        return cls(input_signature=input_signature, layer_specs=layer_specs)

    def to_config(self) -> Dict[str, Any]:
        """Convert the SequentialModel instance to a configuration dictionary."""
        config = {}
        config["layer_specs"] = self.layerspecs
        config["input_signature"] = self.input_signature
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
