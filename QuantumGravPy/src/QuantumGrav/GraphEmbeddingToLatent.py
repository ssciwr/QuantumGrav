from typing import Any, Callable, Sequence
from pathlib import Path
from inspect import isclass
import torch
from . import utils
from . import linear_sequential as QGLS
from . import gnn_block as QGGNN
from .gnn_model import instantiate_type
from . import base
import jsonschema

class GraphEmbeddingToLatent(torch.nn.Module):
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
            "bottleneck": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": [
                            {"description": "bottleneck module type"},
                            {"type": "array", "items": {}},
                            {"type": "object"}
                        ],
                    },
                ]
            },
            "mu_head": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": [
                            {"description": "mu_head module type"},
                            {"type": "array", "items": {}},
                            {"type": "object"}
                        ],
                    },
                ]
            },
            "logvar_head": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": [
                            {"description": "logvar_head module type"},
                            {"type": "array", "items": {}},
                            {"type": "object"}
                        ],
                    },
                ]
            }
        },
        "required": ["pooling_layers", "aggregate_pooling_type"],
        "additionalProperties": False
    }

    def __init__(
        self,
        pooling_layers: Sequence[tuple[type | torch.nn.Module, Sequence[Any] | None, dict[str, Any] | None]] | None = None,
        aggregate_pooling_type: type | torch.nn.Module | Callable | None = None,
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
        Initialize the latent module with pooling optional bottleneck and μ/logvar heads.

        Parameters
        ----------
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

        Notes
        -----
        If both `mu_head_type` and `logvar_head_type` are provided, the module operates in
        VAE mode and performs sampling. Otherwise, it acts as an identity latent module,
        forwarding encoder embeddings directly as latent vectors.
        """
        super().__init__()

        # Pooling layers (required)
        if pooling_layers is None or len(pooling_layers) == 0:
            raise ValueError("LatentModule requires pooling_layers: latent operates on graph-level embeddings.")
        self.pooling_layers = torch.nn.ModuleList(
            [instantiate_type(t, a, k) for (t, a, k) in pooling_layers]
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


    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            h (torch.Tensor): Input tensor representing the graph-level embedding.

        Returns:
            tuple:
                - z (torch.Tensor): Sampled latent vector (or h if no VAE heads are defined).
                - mu (torch.Tensor | None): Mean vector of the latent distribution, or None.
                - logvar (torch.Tensor | None): Log-variance vector of the latent distribution, or None.
        """
        # pooling
        pooled = [pl(h) for pl in self.pooling_layers]
        h = self.aggregate_pooling(pooled)

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
        b_cfg = cfg.get("bottleneck")
        if b_cfg:
            b_type, b_args, b_kwargs = b_cfg
        else:
            b_type = b_args = b_kwargs = None
        m_cfg = cfg.get("mu_head")
        if m_cfg:
            m_type, m_args, m_kwargs = m_cfg
        else:
            m_type = m_args = m_kwargs = None
        l_cfg = cfg.get("logvar_head")
        if l_cfg:
            l_type, l_args, l_kwargs = l_cfg
        else:
            l_type = l_args = l_kwargs = None
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