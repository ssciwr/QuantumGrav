# torch
import torch

from .. import base

# quality of life
from typing import Any
from pathlib import Path
from jsonschema import validate

class NodeUpdateGRU(torch.nn.Module, base.Configurable):
    """
    Node-state update module for autoregressive graph decoding.

    This module updates the hidden state of a newly generated node based on the
    hidden states of its parent nodes. It is designed for decoders that grow
    graphs sequentially and therefore do not require full message passing over
    all edges at each step.

    The update proceeds in two stages:
        1. Aggregation of parent node states into a single vector using a
           permutation-invariant operation (mean, sum, max, or learned pooling).
        2. A GRU-style update that maps the aggregated parent information to the
           hidden state of the new node.

    Formally:
        {h_p}_parents → aggregate → h_agg → GRU → h_new

    where all hidden states live in the same hidden_dim space.

    The module assumes:
        - parent states are already computed and fixed,
        - the aggregation output dimension matches the GRU input dimension,
        - no explicit message passing or edge features are required.

    This design mirrors causal-set sequential growth rules, where a node’s state
    is determined solely by its immediate ancestors rather than by global graph
    updates.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "NodeUpdateGRU Configuration",
        "type": "object",
        "properties": {
            "gru_args": {
                "type": "array",
                "description": "Positional arguments passed to the GRUCell. Must be [input_dim, hidden_dim].",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "integer",
                    "minimum": 1
                    }
            },

            "gru_kwargs": {
                "type": "object",
                "description": "Keyword arguments for the GRUCell.",
                "properties": {
                    "bias": {
                    "type": "boolean"
                    },
                    "device": {
                    "anyOf": [
                        { "type": "null" },
                        { "type": "string" }
                    ]
                    },
                    "dtype": {
                    "anyOf": [
                        { "type": "null" },
                        { "type": "string" }
                    ]
                    }
                },
                "additionalProperties": False
                },

            "aggregation_method": {
                "type": "string",
                "description": "Aggregation method for parent states.",
                "enum": ["mean", "sum", "max", "mlp"]
                },

            "pooling_mlp_type": {
                "description": "MLP used when aggregation='mlp'"
            },
            "pooling_mlp_args": {
                "type": "array",
                "description": "Arguments for pooling MLP",
                "items": {},
            },
            "pooling_mlp_kwargs": {
                "type": "object",
                "description": "Keyword arguments for the pooling MLP",
            }
        },
        "required": ["gru_args"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        gru_args: list[int],
        gru_kwargs: dict[str, Any] | None = None,
        aggregation_method: str = "mean",
        pooling_mlp_type: type[torch.nn.Module] | None = None,
        pooling_mlp_args: list[Any] | None = None,
        pooling_mlp_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a NodeUpdateGRU block.

        Args:
            gru_args : list[Any]
            Positional arguments passed to torch.nn.GRUCell. They
            must be integers specifying:
                gru_args[0] → input_dim
                gru_args[1] → hidden_dim

            Example:
                gru_args = [32, 32]  # input_dim = 32, hidden_dim = 32

            gru_kwargs : dict[str, Any]
            Keyword arguments for the GRU module.

            aggregation_method : str
            One of {"mean", "sum", "max", "mlp"}.

            pooling_mlp_type : type | None
            Optional MLP class used when aggregation_method='mlp'.

            pooling_mlp_args : list[Any] | None
            Positional arguments for the pooling MLP.

            pooling_mlp_kwargs : dict[str, Any] | None
            Keyword arguments for the pooling MLP.
        """

        super(NodeUpdateGRU, self).__init__()

        self.gru_args = gru_args
        self.gru_kwargs = gru_kwargs

        self.aggregation_method = aggregation_method

        self.pooling_mlp_type = pooling_mlp_type
        self.pooling_mlp_args = pooling_mlp_args
        self.pooling_mlp_kwargs = pooling_mlp_kwargs

        # Validate gru_args
        if not isinstance(self.gru_args, (list, tuple)):
            raise ValueError(
                f"gru_args must be a list or tuple of at least two integers "
                f"(input_dim, hidden_dim), but got type {type(self.gru_args)}."
            )

        if self.gru_args is None or len(self.gru_args) < 2:
            raise ValueError(
                "gru_args must specify at least [input_dim, hidden_dim]. "
                "The first two elements of gru_args MUST be integers representing "
                "input_dim and hidden_dim respectively."
            )

        self.in_dim = self.gru_args[0]
        self.hidden_dim = self.gru_args[1]

        # Check integer types
        if not isinstance(self.in_dim, int) or not isinstance(self.hidden_dim, int):
            raise TypeError(
                "gru_args[0] and gru_args[1] must be integers specifying input_dim "
                "and hidden_dim for the GRU module."
            )

        # Check positivity
        if self.in_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                f"input_dim and hidden_dim must be positive integers, but received "
                f"gru_args[0]={self.in_dim}, gru_args[1]={self.hidden_dim}."
            )

        if aggregation_method == "mlp":
            if pooling_mlp_type is None:
                raise ValueError("pooling_mlp_type must be provided when aggregation_method='mlp'.")
            self.pooling_mlp = pooling_mlp_type(
                *(pooling_mlp_args if pooling_mlp_args is not None else []),
                **(pooling_mlp_kwargs if pooling_mlp_kwargs is not None else {})
            )
        else:
            self.pooling_mlp = None

        # GRU
        self.gru = self.torch.nn.GRUCell(
            *gru_args,
            **(gru_kwargs if gru_kwargs is not None else {})
        )

    def aggregate(
        self,
        parent_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate parent hidden states.

        parent_states: Tensor of shape (num_parents, hidden_dim)

        Returns:
            Tensor of shape (hidden_dim,)
        """

        if self.aggregation_method == "mean":
            return parent_states.mean(dim=0)

        if self.aggregation_method == "sum":
            return parent_states.sum(dim=0)

        if self.aggregation_method == "max":
            return parent_states.max(dim=0).values

        if self.aggregation_method == "mlp":
            return self.pooling_mlp(parent_states)

        raise ValueError(f"Unknown aggregation method '{self.aggregation_method}'.")

    # prepare_input method removed; parent_agg is now passed directly to GRU.

    def forward(
        self,
        parent_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full node-state update:
          1. aggregate parent states
          2. GRU update

        parent_states: (num_parents, hidden_dim)
        h_prev:        (hidden_dim,)
        """
        # Empty parent set is invalid: NodeUpdateGRU is only called when parents exist.
        if parent_states.size(0) == 0:
            raise ValueError(
                "NodeUpdateGRU.forward() received zero parent states, "
                "but at least one parent is required."
            )
        # Dimensionality mismatch check
        if parent_states.size(1) != self.in_dim:
            raise ValueError(
                f"NodeUpdateGRU.forward() received parent_states with wrong feature dimension: "
                f"expected {self.in_dim}, got {parent_states.size(1)}."
            )
        parent_agg = self.aggregate(parent_states)
        return self.gru(parent_agg)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NodeUpdateGRU":
        validate(config, cls.schema)
        return cls(
            gru_args=config.get("gru_args", []),
            gru_kwargs=config.get("gru_kwargs", {}),
            aggregation_method=config.get("aggregation_method", "mean"),
            pooling_mlp_type=config.get("pooling_mlp_type", None),
            pooling_mlp_args=config.get("pooling_mlp_args", []),
            pooling_mlp_kwargs=config.get("pooling_mlp_kwargs", {}),
        )


    def to_config(self) -> dict[str, Any]:
        return {
            "gru_args": self.gru_args if self.gru_args is not None else [],
            "gru_kwargs": self.gru_kwargs if self.gru_kwargs is not None else {},
            "aggregation_method": self.aggregation_method,
            "pooling_mlp_type": self.pooling_mlp_type,
            "pooling_mlp_args": self.pooling_mlp_args if self.pooling_mlp_args is not None else [],
            "pooling_mlp_kwargs": self.pooling_mlp_kwargs if self.pooling_mlp_kwargs is not None else {},
        }

    def save(self, path: str | Path) -> None:
        """Save NodeUpdateGRU to a file."""
        cfg = self.to_config()
        torch.save({"config": cfg, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device = torch.device("cpu")) -> "NodeUpdateGRU":
        """Load NodeUpdateGRU from a file."""
        modeldata = torch.load(path, weights_only=False)
        model = cls.from_config(modeldata["config"]).to(device)
        model.load_state_dict(modeldata["state_dict"])
        return model