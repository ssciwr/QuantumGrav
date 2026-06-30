from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from jsonschema import validate

from .. import base
from .. import utils


def _resolve_torch_module(
    module_or_type: str | type[torch.nn.Module],
) -> type[torch.nn.Module]:
    if isinstance(module_or_type, str):
        if hasattr(torch.nn, module_or_type):
            module_or_type = getattr(torch.nn, module_or_type)
        else:
            module_or_type = utils.import_and_get(module_or_type)

    if not isinstance(module_or_type, type) or not issubclass(
        module_or_type,
        torch.nn.Module,
    ):
        raise ValueError(f"{module_or_type} must be a torch.nn.Module type.")

    return module_or_type


def _module_path(module: type[torch.nn.Module]) -> str:
    return f"{module.__module__}.{module.__name__}"


class ObservableCNNEncoder(torch.nn.Module, base.Configurable):
    """Encode named vector observables with configurable 1D CNN branches.

    The encoder is intended for observable-only causal-set models. It builds one
    Conv1d branch for each configured vector input, pools every branch to a
    fixed-size graph-level vector, optionally encodes scalar graph features with
    an MLP, and concatenates all embeddings.
    """

    vector_branch_schema = {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "description": (
                    "Conv1d channel sizes for this branch, including input channel."
                ),
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
            },
            "kernel_sizes": {
                "type": "array",
                "description": "Conv1d kernel sizes for this branch.",
                "items": {"type": "integer", "minimum": 1},
            },
            "paddings": {
                "type": "array",
                "description": (
                    "Conv1d paddings for this branch. Defaults to same-length padding."
                ),
                "items": {"type": "integer", "minimum": 0},
            },
            "activation": {
                "description": (
                    "Activation module type or torch.nn module name for this branch."
                ),
            },
            "activation_kwargs": {
                "type": "object",
                "description": (
                    "Keyword arguments passed to this branch's activation modules."
                ),
            },
            "dropout": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Dropout probability after each branch activation.",
            },
            "pooling": {
                "type": "string",
                "enum": ["avg", "max", "avgmax"],
                "description": "Global pooling type for this branch.",
            },
        },
        "required": ["channels"],
        "additionalProperties": False,
    }

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ObservableCNNEncoder Configuration",
        "type": "object",
        "properties": {
            "vector_inputs": {
                "type": "object",
                "description": (
                    "Mapping from vector input name to Conv1d branch configuration."
                ),
                "additionalProperties": vector_branch_schema,
                "minProperties": 1,
            },
            "activation": {
                "description": (
                    "Default activation module type or torch.nn module name for vector "
                    "branches."
                ),
            },
            "activation_kwargs": {
                "type": "object",
                "description": (
                    "Default keyword arguments passed to vector branch activation "
                    "modules."
                ),
            },
            "dropout": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Default dropout probability after each vector branch activation."
                ),
            },
            "pooling": {
                "type": "string",
                "enum": ["avg", "max", "avgmax"],
                "description": "Default global pooling type for vector branches.",
            },
            "scalar_in_features": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of scalar input features.",
            },
            "scalar_hidden_dims": {
                "type": "array",
                "description": "Hidden/output dimensions for the scalar-feature MLP.",
                "items": {"type": "integer", "minimum": 1},
            },
            "scalar_activation": {
                "description": (
                    "Activation module type or torch.nn module name for scalar MLP "
                    "hidden layers."
                ),
            },
            "scalar_activation_kwargs": {
                "type": "object",
                "description": (
                    "Keyword arguments passed to scalar MLP activation modules."
                ),
            },
        },
        "required": ["vector_inputs"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        vector_inputs: Mapping[str, Mapping[str, Any]],
        activation: str | type[torch.nn.Module] = torch.nn.ReLU,
        activation_kwargs: Dict[str, Any] | None = None,
        dropout: float = 0.0,
        pooling: str = "avg",
        scalar_in_features: int = 0,
        scalar_hidden_dims: Sequence[int] | None = None,
        scalar_activation: str | type[torch.nn.Module] = torch.nn.ReLU,
        scalar_activation_kwargs: Dict[str, Any] | None = None,
    ):
        """Create an observable CNN encoder.

        Args:
            vector_inputs: Mapping from vector input name to branch config. Each
                branch config must provide ``channels`` and can override
                ``kernel_sizes``, ``paddings``, ``activation``,
                ``activation_kwargs``, ``dropout``, and ``pooling``.
            activation: Default activation module type or torch.nn module name
                used in vector branches.
            activation_kwargs: Default keyword arguments for vector activations.
            dropout: Default dropout probability after each vector activation.
            pooling: Default global pooling mode: ``"avg"``, ``"max"``, or ``"avgmax"``.
            scalar_in_features: Number of scalar features supplied to ``forward``.
            scalar_hidden_dims: Dimensions for the scalar-feature MLP. If empty,
                scalar features are passed through unchanged.
            scalar_activation: Activation module type or torch.nn module name used
                between scalar MLP layers.
            scalar_activation_kwargs: Keyword arguments for scalar activations.
        """
        super().__init__()

        if len(vector_inputs) == 0:
            raise ValueError("vector_inputs must contain at least one branch.")

        if pooling not in {"avg", "max", "avgmax"}:
            raise ValueError("pooling must be one of 'avg', 'max', or 'avgmax'.")

        if scalar_in_features < 0:
            raise ValueError("scalar_in_features must be non-negative.")

        self.activation = _resolve_torch_module(activation)
        self.activation_kwargs = activation_kwargs or {}
        self.dropout = dropout
        self.pooling = pooling
        self.scalar_in_features = scalar_in_features
        self.scalar_hidden_dims = list(scalar_hidden_dims or [])
        self.scalar_activation = _resolve_torch_module(scalar_activation)
        self.scalar_activation_kwargs = scalar_activation_kwargs or {}

        self.vector_inputs = self._normalize_vector_inputs(vector_inputs)
        self.vector_encoders = torch.nn.ModuleDict(
            {
                name: self._build_conv_branch(
                    config["channels"],
                    config["kernel_sizes"],
                    config["paddings"],
                    config["activation"],
                    config["activation_kwargs"],
                    config["dropout"],
                )
                for name, config in self.vector_inputs.items()
            }
        )
        self.scalar_encoder = self._build_scalar_branch()

        self.vector_out_features = {
            name: self._branch_out_features(config)
            for name, config in self.vector_inputs.items()
        }
        self.scalar_out_features = (
            self.scalar_hidden_dims[-1]
            if len(self.scalar_hidden_dims) > 0
            else self.scalar_in_features
        )
        self.out_features = (
            sum(self.vector_out_features.values()) + self.scalar_out_features
        )

    def _normalize_vector_inputs(
        self,
        vector_inputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for name, config in vector_inputs.items():
            if "." in name:
                raise ValueError("vector input names cannot contain '.'.")

            channels = list(config["channels"])
            kernel_sizes = self._default_kernel_sizes(
                config.get("kernel_sizes"),
                channels,
            )
            paddings = self._default_paddings(config.get("paddings"), kernel_sizes)
            branch_activation = _resolve_torch_module(
                config.get("activation", self.activation)
            )
            branch_activation_kwargs = config.get(
                "activation_kwargs",
                self.activation_kwargs,
            )
            branch_dropout = config.get("dropout", self.dropout)
            branch_pooling = config.get("pooling", self.pooling)
            if branch_pooling not in {"avg", "max", "avgmax"}:
                raise ValueError(
                    f"pooling for vector input '{name}' must be one of "
                    "'avg', 'max', or 'avgmax'."
                )

            normalized[name] = {
                "channels": channels,
                "kernel_sizes": kernel_sizes,
                "paddings": paddings,
                "activation": branch_activation,
                "activation_kwargs": branch_activation_kwargs,
                "dropout": branch_dropout,
                "pooling": branch_pooling,
            }

        return normalized

    @staticmethod
    def _default_kernel_sizes(
        kernel_sizes: Sequence[int] | None,
        channels: Sequence[int],
    ) -> list[int]:
        num_convs = len(channels) - 1
        if num_convs < 1:
            raise ValueError("channels must contain at least input and output size.")

        if kernel_sizes is None:
            return [5 for _ in range(num_convs)]

        kernel_sizes = list(kernel_sizes)
        if len(kernel_sizes) != num_convs:
            raise ValueError("kernel_sizes must have length len(channels) - 1.")
        return kernel_sizes

    @staticmethod
    def _default_paddings(
        paddings: Sequence[int] | None,
        kernel_sizes: Sequence[int],
    ) -> list[int]:
        if paddings is None:
            return [kernel_size // 2 for kernel_size in kernel_sizes]

        paddings = list(paddings)
        if len(paddings) != len(kernel_sizes):
            raise ValueError("paddings must have the same length as kernel_sizes.")
        return paddings

    @staticmethod
    def _branch_out_features(config: Mapping[str, Any]) -> int:
        pooling_multiplier = 2 if config["pooling"] == "avgmax" else 1
        return config["channels"][-1] * pooling_multiplier

    def _build_conv_branch(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        paddings: Sequence[int],
        activation: type[torch.nn.Module],
        activation_kwargs: Mapping[str, Any],
        dropout: float,
    ) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(
                torch.nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                )
            )
            layers.append(activation(**activation_kwargs))
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

        return torch.nn.Sequential(*layers)

    def _build_scalar_branch(self) -> torch.nn.Module:
        if self.scalar_in_features == 0:
            return torch.nn.Identity()

        if len(self.scalar_hidden_dims) == 0:
            return torch.nn.Identity()

        layers: list[torch.nn.Module] = []
        dims = [self.scalar_in_features, *self.scalar_hidden_dims]
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(self.scalar_activation(**self.scalar_activation_kwargs))

        return torch.nn.Sequential(*layers)

    @staticmethod
    def _as_conv_input(signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim == 1:
            return signal.unsqueeze(0).unsqueeze(0)
        if signal.ndim == 2:
            return signal.unsqueeze(1)
        if signal.ndim == 3:
            return signal
        raise ValueError(
            "Vector inputs must have shape [length], [batch, length], "
            f"or [batch, channels, length], got {tuple(signal.shape)}."
        )

    @staticmethod
    def _as_scalar_input(scalars: torch.Tensor) -> torch.Tensor:
        if scalars.ndim == 1:
            return scalars.unsqueeze(0)
        if scalars.ndim == 2:
            return scalars
        raise ValueError(
            "Scalar features must have shape [features] or [batch, features], "
            f"got {tuple(scalars.shape)}."
        )

    @staticmethod
    def _pool(x: torch.Tensor, pooling: str) -> torch.Tensor:
        if pooling == "avg":
            return torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        if pooling == "max":
            return torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)

        avg = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_ = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)
        return torch.cat([avg, max_], dim=-1)

    def forward(
        self,
        vector_features: Mapping[str, torch.Tensor],
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode configured vector and scalar inputs into one graph-level embedding.

        Args:
            vector_features: Mapping from configured vector input name to tensor.
                Each tensor must have shape ``[length]``, ``[batch, length]``, or
                ``[batch, channels, length]``.
            scalar_features: Optional scalar feature tensor with shape
                ``[features]`` or ``[batch, features]``.

        Returns:
            torch.Tensor: Concatenated embedding of shape ``[batch, out_features]``.
        """
        missing_inputs = set(self.vector_inputs) - set(vector_features)
        if missing_inputs:
            raise ValueError(f"Missing vector inputs: {sorted(missing_inputs)}.")

        embeddings = []
        for name, encoder in self.vector_encoders.items():
            branch_input = self._as_conv_input(vector_features[name])
            branch_embedding = self._pool(
                encoder(branch_input),
                self.vector_inputs[name]["pooling"],
            )
            embeddings.append(branch_embedding)

        if self.scalar_in_features > 0:
            if scalar_features is None:
                raise ValueError(
                    "scalar_features must be supplied when scalar_in_features > 0."
                )
            scalar_features = self._as_scalar_input(scalar_features)
            embeddings.append(self.scalar_encoder(scalar_features))

        return torch.cat(embeddings, dim=-1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ObservableCNNEncoder":
        """Construct an ObservableCNNEncoder from a configuration dictionary."""
        validate(config, cls.schema)

        try:
            return cls(
                vector_inputs=config["vector_inputs"],
                activation=config.get("activation", torch.nn.ReLU),
                activation_kwargs=config.get("activation_kwargs", None),
                dropout=config.get("dropout", 0.0),
                pooling=config.get("pooling", "avg"),
                scalar_in_features=config.get("scalar_in_features", 0),
                scalar_hidden_dims=config.get("scalar_hidden_dims", None),
                scalar_activation=config.get("scalar_activation", torch.nn.ReLU),
                scalar_activation_kwargs=config.get("scalar_activation_kwargs", None),
            )
        except Exception as e:
            raise RuntimeError(
                f"Error while building ObservableCNNEncoder from config: {e}"
            ) from e

    def to_config(self) -> Dict[str, Any]:
        """Build a serializable configuration dictionary for this encoder."""
        vector_inputs = {
            name: {
                "channels": config["channels"],
                "kernel_sizes": config["kernel_sizes"],
                "paddings": config["paddings"],
                "activation": _module_path(config["activation"]),
                "activation_kwargs": config["activation_kwargs"],
                "dropout": config["dropout"],
                "pooling": config["pooling"],
            }
            for name, config in self.vector_inputs.items()
        }
        return {
            "vector_inputs": vector_inputs,
            "activation": _module_path(self.activation),
            "activation_kwargs": self.activation_kwargs,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "scalar_in_features": self.scalar_in_features,
            "scalar_hidden_dims": self.scalar_hidden_dims,
            "scalar_activation": _module_path(self.scalar_activation),
            "scalar_activation_kwargs": self.scalar_activation_kwargs,
        }

    def save(self, path: str | Path) -> None:
        """Save the encoder config and state dict to disk."""
        torch.save({"config": self.to_config(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> "ObservableCNNEncoder":
        """Load an ObservableCNNEncoder from a file produced by ``save``."""
        payload = torch.load(path, map_location=device)
        model = cls.from_config(payload["config"])
        model.load_state_dict(payload["state_dict"], strict=False)
        model.to(device)
        return model
