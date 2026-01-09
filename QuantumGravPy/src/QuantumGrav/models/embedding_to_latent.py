from typing import Any, Callable, Sequence
import torch
from ..gnn_model import instantiate_type
from .. import base
import jsonschema

class GraphEmbeddingToLatent(torch.nn.Module, base.Configurable):
    """
    Module that transforms graph-level embeddings into latent representations.

    This module operates after a GNN encoder has produced node-level embeddings.
    It performs mandatory pooling to obtain a graph-level vector, optionally
    applies a bottleneck transformation, and—if VAE heads are provided—produces
    μ and log σ² for stochastic latent sampling.

    The latent module therefore implements:
        node embeddings → pooling → graph embedding → (optional bottleneck)
             → (optional μ/logvar heads) → latent vector

    Pooling is mandatory because the latent representation must be defined at
    the graph level, not per node. Without pooling, dimensionalities would be
    inconsistent.
    """

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "LatentModule Configuration",
        "type": "object",
        "properties": {
            "pooling_layers": {
                "type": "array",
                "description": "MANDATORY: pooling layers are required because the latent operates on graph-level embeddings. Each entry is [type, args, kwargs].",
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": [
                        {"description": "Pooling type (class or module)"},
                        {"type": "array", "items": {}},
                        {"type": "object"}
                    ]
                }
            },
            "aggregate_pooling_type": {
                "description": "MANDATORY: aggregation operation for pooling outputs."
            },
            "aggregate_pooling_args": {
                "type": "array",
                "items": {}
            },
            "aggregate_pooling_kwargs": {
                "type": "object"
            },
            "bottleneck_type": {
                "description": "Optional: class/module for bottleneck transformation. If omitted, no bottleneck is applied."
            },
            "bottleneck_args": {
                "type": ["array", "null"],
                "description": "Positional arguments for bottleneck_type.",
                "items": {}
            },
            "bottleneck_kwargs": {
                "type": ["object", "null"],
                "description": "Keyword arguments for bottleneck_type."
            },
            "mu_head_type": {
                "description": "Optional: class/module producing the latent mean μ. Must be provided together with logvar head."
            },
            "mu_head_args": {
                "type": ["array", "null"],
                "description": "Positional arguments for mu_head_type.",
                "items": {}
            },
            "mu_head_kwargs": {
                "type": ["object", "null"],
                "description": "Keyword arguments for mu_head_type."
            },
            "logvar_head_type": {
                "description": "Optional: class/module producing latent log-variance log σ². Must be provided together with mu head."
            },
            "logvar_head_args": {
                "type": ["array", "null"],
                "description": "Positional arguments for logvar_head_type.",
                "items": {}
            },
            "logvar_head_kwargs": {
                "type": ["object", "null"],
                "description": "Keyword arguments for logvar_head_type."
            },
        },
        "required": ["pooling_layers", "aggregate_pooling_type"],
        "additionalProperties": False
    }

    def __init__(
        self,
        pooling_layers: Sequence[tuple[type | torch.nn.Module, Sequence[Any] | None, dict[str, Any] | None]],
        aggregate_pooling_type: type | torch.nn.Module | Callable,
        aggregate_pooling_args: Sequence[Any] | None = None,
        aggregate_pooling_kwargs: dict[str, Any] | None = None,
        bottleneck_type: type | torch.nn.Module | None = None,
        bottleneck_args: Sequence[Any] | None = None,
        bottleneck_kwargs: dict[str, Any] | None = None,
        mu_head_type: type | torch.nn.Module | None = None,
        mu_head_args: Sequence[Any] | None = None,
        mu_head_kwargs: dict[str, Any] | None = None,
        logvar_head_type: type | torch.nn.Module | None = None,
        logvar_head_args: Sequence[Any] | None = None,
        logvar_head_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a graph-embeddings-to-latent module.

        Args:
            pooling_layers : Sequence[tuple[type | torch.nn.Module, Sequence[Any] | None, dict[str, Any] | None]]
            Mandatory sequence of pooling layers. Each tuple is (type, args, kwargs). Pooling converts node-level
            embeddings into a graph-level embedding.

            aggregate_pooling_type : type | torch.nn.Module | Callable
            Mandatory module or callable that aggregates the outputs of all pooling layers into a single embedding.

            aggregate_pooling_args : Sequence[Any] | None
            Positional arguments for the aggregate pooling constructor.

            aggregate_pooling_kwargs : dict[str, Any] | None
            Keyword arguments for the aggregate pooling constructor.

            bottleneck_type : type | torch.nn.Module | None
            Class or module used to reduce encoder output before producing mu/logvar.
            bottleneck_args : Sequence[Any] | None
            Positional arguments passed to `bottleneck_type`.
            bottleneck_kwargs : dict[str, Any] | None
            Keyword arguments passed to `bottleneck_type`.

            mu_head_type : type | torch.nn.Module | None
            Class or module that maps the bottleneck output to the latent mean vector μ.
            mu_head_args : Sequence[Any] | None
            Positional arguments for `mu_head_type`.
            mu_head_kwargs : dict[str, Any] | None
            Keyword arguments for `mu_head_type`.

            logvar_head_type : type | torch.nn.Module | None
            Class or module that maps the bottleneck output to the latent log-variance vector log σ².
            logvar_head_args : Sequence[Any] | None
            Positional arguments for `logvar_head_type`.
            logvar_head_kwargs : dict[str, Any] | None
            Keyword arguments for `logvar_head_type`.
        """
        super().__init__()

        # Pooling layers (required)
        if pooling_layers is None or len(pooling_layers) == 0:
            raise ValueError("LatentModule requires pooling_layers: latent operates on graph-level embeddings.")
        self.pooling_layers = torch.nn.ModuleList(
            [instantiate_type(pl_type, args, kwargs) for (pl_type, args, kwargs) in pooling_layers]
        )
        if aggregate_pooling_type is None:
            raise ValueError("LatentModule requires aggregate_pooling_type; pooling is mandatory for graph-level latent vectors.")
        self.aggregate_pooling = instantiate_type(
            aggregate_pooling_type,
            aggregate_pooling_args,
            aggregate_pooling_kwargs
        )

        self.bottleneck = instantiate_type(bottleneck_type, bottleneck_args, bottleneck_kwargs) if bottleneck_type is not None else None
        self.mu_head = instantiate_type(mu_head_type, mu_head_args, mu_head_kwargs) if mu_head_type is not None else None
        self.logvar_head = instantiate_type(logvar_head_type, logvar_head_args, logvar_head_kwargs) if logvar_head_type is not None else None

        # Consistency checks: forbid only one of mu_head/logvar_head
        if (self.mu_head is None) ^ (self.logvar_head is None):
            raise ValueError(
                "LatentModule misconfigured: either both mu_head and logvar_head must be provided (VAE mode), "
                "or neither (non‑VAE mode). Providing only one is not supported."
            )


    def forward(self, embeddings: torch.Tensor, batch: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            embeddings (torch.Tensor): Input tensor representing the graph-level embedding.

        Returns:
            tuple:
                - z (torch.Tensor): Sampled latent vector (or h if no VAE heads are defined).
                - mu (torch.Tensor | None): Mean vector of the latent distribution, or None.
                - logvar (torch.Tensor | None): Log-variance vector of the latent distribution, or None.
        """
        # pooling
        pooled_embeddings = [
                    pooling_op(embeddings, batch)
                    for pooling_op in self.pooling_layers
                ]

        h = self.aggregate_pooling(pooled_embeddings)

        if self.bottleneck is not None:
            h = self.bottleneck(h)

        if self.mu_head is None or self.logvar_head is None:
            return h, None, None
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        z = mu + eps * std
        return z, mu, logvar


    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LatentModule":
        jsonschema.validate(cfg, cls.schema)
        b_type = cfg.get("bottleneck_type")
        b_args = cfg.get("bottleneck_args")
        b_kwargs = cfg.get("bottleneck_kwargs")
        m_type = cfg.get("mu_head_type")
        m_args = cfg.get("mu_head_args")
        m_kwargs = cfg.get("mu_head_kwargs")
        l_type = cfg.get("logvar_head_type")
        l_args = cfg.get("logvar_head_args")
        l_kwargs = cfg.get("logvar_head_kwargs")
        pl = cfg.get("pooling_layers")
        ag_type = cfg.get("aggregate_pooling_type")
        ag_args = cfg.get("aggregate_pooling_args")
        ag_kwargs = cfg.get("aggregate_pooling_kwargs")
        return cls(
            pooling_layers=pl,
            aggregate_pooling_type=ag_type,
            aggregate_pooling_args=ag_args,
            aggregate_pooling_kwargs=ag_kwargs,
            bottleneck_type=b_type, bottleneck_args=b_args, bottleneck_kwargs=b_kwargs,
            mu_head_type=m_type, mu_head_args=m_args, mu_head_kwargs=m_kwargs,
            logvar_head_type=l_type, logvar_head_args=l_args, logvar_head_kwargs=l_kwargs,
        )