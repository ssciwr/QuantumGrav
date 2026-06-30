from __future__ import annotations

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
        if not hasattr(torch.nn, module_or_type):
            raise ValueError(f"Unknown torch.nn module '{module_or_type}'.")
        module_or_type = getattr(torch.nn, module_or_type)

    if not isinstance(module_or_type, type) or not issubclass(
        module_or_type,
        torch.nn.Module,
    ):
        raise ValueError(f"{module_or_type} must be a torch.nn.Module type.")

    return module_or_type


def _module_path(module: type[torch.nn.Module]) -> str:
    return f"{module.__module__}.{module.__name__}"


class ObservableCNNEncoder(torch.nn.Module, base.Configurable):
    """Encode graph-observable signals with separate 1D CNN branches.

    The encoder is intended for observable-only causal-set classifiers. It takes a
    probability-normalized link-degree signal, a preprocessed interval-abundance
    signal, and optional scalar graph observables such as Laplacian eigenvalues.
    The two 1D signals are encoded independently, pooled to fixed-size vectors,
    concatenated with a scalar-feature MLP, and returned as a graph-level
    embedding. A downstream task head can then map the embedding to logits.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ObservableCNNEncoder Configuration",
        "type": "object",
        "properties": {
            "degree_channels": {
                "type": "array",
                "description": "Conv1d channel sizes for the degree branch, including input channel.",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
            },
            "interval_channels": {
                "type": "array",
                "description": "Conv1d channel sizes for the interval branch, including input channel.",
                "items": {"type": "integer", "minimum": 1},
                "minItems": 2,
            },
            "degree_kernel_sizes": {
                "type": "array",
                "description": "Conv1d kernel sizes for the degree branch.",
                "items": {"type": "integer", "minimum": 1},
            },
            "interval_kernel_sizes": {
                "type": "array",
                "description": "Conv1d kernel sizes for the interval branch.",
                "items": {"type": "integer", "minimum": 1},
            },
            "degree_paddings": {
                "type": "array",
                "description": "Conv1d paddings for the degree branch. Defaults to same-length padding.",
                "items": {"type": "integer", "minimum": 0},
            },
            "interval_paddings": {
                "type": "array",
                "description": "Conv1d paddings for the interval branch. Defaults to same-length padding.",
                "items": {"type": "integer", "minimum": 0},
            },
            "activation": {
                "description": "Activation module type or torch.nn module name.",
            },
            "activation_kwargs": {
                "type": "object",
                "description": "Keyword arguments passed to each activation module.",
            },
            "dropout": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Dropout probability after each activation. Set to 0 to disable.",
            },
            "pooling": {
                "type": "string",
                "enum": ["avg", "max", "avgmax"],
                "description": "Global pooling type for each 1D branch.",
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
                "description": "Activation module type or torch.nn module name for scalar MLP hidden layers.",
            },
            "scalar_activation_kwargs": {
                "type": "object",
                "description": "Keyword arguments passed to scalar MLP activation modules.",
            },
        },
        "required": [
            "degree_channels",
            "interval_channels",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        degree_channels: Sequence[int],
        interval_channels: Sequence[int],
        degree_kernel_sizes: Sequence[int] | None = None,
        interval_kernel_sizes: Sequence[int] | None = None,
        degree_paddings: Sequence[int] | None = None,
        interval_paddings: Sequence[int] | None = None,
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
            degree_channels: Channel sizes for the degree Conv1d branch, including
                the input channel. For a single input signal this should start with 1.
            interval_channels: Channel sizes for the interval Conv1d branch, including
                the input channel. For a single input signal this should start with 1.
            degree_kernel_sizes: Kernel sizes for the degree branch. Defaults to 5
                for every convolution.
            interval_kernel_sizes: Kernel sizes for the interval branch. Defaults to
                5 for every convolution.
            degree_paddings: Paddings for the degree branch. Defaults to
                ``kernel_size // 2`` for each convolution.
            interval_paddings: Paddings for the interval branch. Defaults to
                ``kernel_size // 2`` for each convolution.
            activation: Activation module type or torch.nn module name used in CNN branches.
            activation_kwargs: Keyword arguments for CNN activations.
            dropout: Dropout probability after each CNN activation.
            pooling: Global pooling mode: ``"avg"``, ``"max"``, or ``"avgmax"``.
            scalar_in_features: Number of scalar features supplied to ``forward``.
            scalar_hidden_dims: Dimensions for the scalar-feature MLP. If empty,
                scalar features are passed through unchanged.
            scalar_activation: Activation module type or torch.nn module name used
                between scalar MLP layers.
            scalar_activation_kwargs: Keyword arguments for scalar activations.
        """
        super().__init__()

        if pooling not in {"avg", "max", "avgmax"}:
            raise ValueError("pooling must be one of 'avg', 'max', or 'avgmax'.")

        if scalar_in_features < 0:
            raise ValueError("scalar_in_features must be non-negative.")

        self.degree_channels = list(degree_channels)
        self.interval_channels = list(interval_channels)
        self.degree_kernel_sizes = self._default_kernel_sizes(
            degree_kernel_sizes, self.degree_channels
        )
        self.interval_kernel_sizes = self._default_kernel_sizes(
            interval_kernel_sizes, self.interval_channels
        )
        self.degree_paddings = self._default_paddings(
            degree_paddings, self.degree_kernel_sizes
        )
        self.interval_paddings = self._default_paddings(
            interval_paddings, self.interval_kernel_sizes
        )
        self.activation = _resolve_torch_module(activation)
        self.activation_kwargs = activation_kwargs or {}
        self.dropout = dropout
        self.pooling = pooling
        self.scalar_in_features = scalar_in_features
        self.scalar_hidden_dims = list(scalar_hidden_dims or [])
        self.scalar_activation = _resolve_torch_module(scalar_activation)
        self.scalar_activation_kwargs = scalar_activation_kwargs or {}

        self.degree_encoder = self._build_conv_branch(
            self.degree_channels,
            self.degree_kernel_sizes,
            self.degree_paddings,
        )
        self.interval_encoder = self._build_conv_branch(
            self.interval_channels,
            self.interval_kernel_sizes,
            self.interval_paddings,
        )
        self.scalar_encoder = self._build_scalar_branch()

        pooling_multiplier = 2 if self.pooling == "avgmax" else 1
        self.degree_out_features = self.degree_channels[-1] * pooling_multiplier
        self.interval_out_features = self.interval_channels[-1] * pooling_multiplier
        self.scalar_out_features = (
            self.scalar_hidden_dims[-1]
            if len(self.scalar_hidden_dims) > 0
            else self.scalar_in_features
        )
        self.out_features = (
            self.degree_out_features
            + self.interval_out_features
            + self.scalar_out_features
        )

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

    def _build_conv_branch(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        paddings: Sequence[int],
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
            layers.append(self.activation(**self.activation_kwargs))
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(p=self.dropout))

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
            "Observable signals must have shape [length], [batch, length], "
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

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "avg":
            return torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        if self.pooling == "max":
            return torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)

        avg = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        max_ = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)
        return torch.cat([avg, max_], dim=-1)

    def forward(
        self,
        degree_distribution: torch.Tensor,
        interval_abundance: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observable tensors into one graph-level embedding.

        Args:
            degree_distribution: Link-degree distribution signal with shape
                ``[length]``, ``[batch, length]``, or ``[batch, channels, length]``.
            interval_abundance: Interval-abundance signal with shape ``[length]``,
                ``[batch, length]``, or ``[batch, channels, length]``.
            scalar_features: Optional scalar feature tensor with shape
                ``[features]`` or ``[batch, features]``.

        Returns:
            torch.Tensor: Concatenated embedding of shape ``[batch, out_features]``.
        """
        degree_distribution = self._as_conv_input(degree_distribution)
        interval_abundance = self._as_conv_input(interval_abundance)

        degree_embedding = self._pool(self.degree_encoder(degree_distribution))
        interval_embedding = self._pool(self.interval_encoder(interval_abundance))

        embeddings = [degree_embedding, interval_embedding]
        if self.scalar_in_features > 0:
            if scalar_features is None:
                raise ValueError("scalar_features must be supplied when scalar_in_features > 0.")
            scalar_features = self._as_scalar_input(scalar_features)
            embeddings.append(self.scalar_encoder(scalar_features))

        return torch.cat(embeddings, dim=-1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ObservableCNNEncoder":
        """Construct an ObservableCNNEncoder from a configuration dictionary."""
        validate(config, cls.schema)

        try:
            return cls(
                degree_channels=config["degree_channels"],
                interval_channels=config["interval_channels"],
                degree_kernel_sizes=config.get("degree_kernel_sizes", None),
                interval_kernel_sizes=config.get("interval_kernel_sizes", None),
                degree_paddings=config.get("degree_paddings", None),
                interval_paddings=config.get("interval_paddings", None),
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
        return {
            "degree_channels": self.degree_channels,
            "interval_channels": self.interval_channels,
            "degree_kernel_sizes": self.degree_kernel_sizes,
            "interval_kernel_sizes": self.interval_kernel_sizes,
            "degree_paddings": self.degree_paddings,
            "interval_paddings": self.interval_paddings,
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
        config = payload["config"]

        if isinstance(config.get("activation"), str):
            config["activation"] = utils.import_and_get(config["activation"])

        if isinstance(config.get("scalar_activation"), str):
            config["scalar_activation"] = utils.import_and_get(
                config["scalar_activation"]
            )

        model = cls.from_config(config)
        model.load_state_dict(payload["state_dict"], strict=False)
        model.to(device)
        return model
