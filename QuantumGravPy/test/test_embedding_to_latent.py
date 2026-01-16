import QuantumGrav as QG
import pytest

import jsonschema

import torch
import torch_geometric

from QuantumGrav.models.embedding_to_latent import GraphEmbeddingToLatent

# -------------------------------------------------------------------------
# Fixtures: pooling + aggregator
# -------------------------------------------------------------------------

@pytest.fixture
def pooling_specs():
    return [
        [torch_geometric.nn.global_mean_pool, [], {}],
        [torch_geometric.nn.global_max_pool, [], {}],
    ]


@pytest.fixture
def aggregate_type():
    return lambda seq: torch.cat(seq, dim=1)


@pytest.fixture
def aggregate_args():
    return []


@pytest.fixture
def aggregate_kwargs():
    return {}


# -------------------------------------------------------------------------
# Fixtures: multistep bottleneck and heads using LinearSequential
# -------------------------------------------------------------------------

@pytest.fixture
def bottleneck_type():
    return QG.models.LinearSequential


@pytest.fixture
def bottleneck_args():
    return [
        [
            [64, 64],  
            [64, 16],  
        ]
    ]

@pytest.fixture
def bottleneck_kwargs():
    return {
        "activations": [torch.nn.ReLU, torch.nn.ReLU],
        "linear_kwargs": [{"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}]
    }


@pytest.fixture
def mu_head_type():
    return QG.models.LinearSequential


@pytest.fixture
def mu_head_args():
    return [
        [
            [16, 16]
        ]
    ]

@pytest.fixture
def mu_head_args_no_bottle():
    return [
        [
            [64, 16]   # instead of [16, 16]
        ]
    ]

@pytest.fixture
def mu_head_kwargs():
    return {
        "activations": [torch.nn.Identity],                  # one layer → one activation
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


@pytest.fixture
def logvar_head_type():
    return QG.models.LinearSequential


@pytest.fixture
def logvar_head_args():
    return [
        [
            [16, 16]
        ]
    ]

@pytest.fixture
def logvar_head_args_no_bottle():
    return [
        [
            [64, 16]   # instead of [16, 16]
        ]
    ]

@pytest.fixture
def logvar_head_kwargs():
    return {
        "activations": [torch.nn.Identity],
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


# -------------------------------------------------------------------------
# Fixtures: input
# -------------------------------------------------------------------------

@pytest.fixture
def node_embeddings():
    return torch.randn(10, 32)


@pytest.fixture
def batch_vector():
    return torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# -------------------------------------------------------------------------
# Fixtures: config
# -------------------------------------------------------------------------

@pytest.fixture
def latent_config(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs
):
    return {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "bottleneck_type": bottleneck_type,
        "bottleneck_args": bottleneck_args,
        "bottleneck_kwargs": bottleneck_kwargs,
        "mu_head_type": mu_head_type,
        "mu_head_args": mu_head_args,
        "mu_head_kwargs": mu_head_kwargs,
        "logvar_head_type": logvar_head_type,
        "logvar_head_args": logvar_head_args,
        "logvar_head_kwargs": logvar_head_kwargs,
    }


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Construction tests
# -------------------------------------------------------------------------


def test_latent_multilayer_construction(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs
):
    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=bottleneck_type,
        bottleneck_args=bottleneck_args,
        bottleneck_kwargs=bottleneck_kwargs,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    assert latent.bottleneck is not None
    assert latent.mu_head is not None
    assert latent.logvar_head is not None
    assert isinstance(latent.pooling_layers, torch.nn.ModuleList)
    assert len(latent.pooling_layers) == 2

def test_construction_with_single_pooling_layer(
    aggregate_type, aggregate_args, aggregate_kwargs,
    node_embeddings, batch_vector
):
    """
    A latent module with exactly one pooling layer must work and produce an
    output equal to the pooling layer's output dimension (here 32).
    """

    # One pooling layer: global_mean_pool
    pooling_specs = [
        [torch_geometric.nn.global_mean_pool, [], {}],
    ]

    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,       # typically concat; single entry → identity
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        mu_head_type=None,
        mu_head_args=None,
        mu_head_kwargs=None,
        logvar_head_type=None,
        logvar_head_args=None,
        logvar_head_kwargs=None,
    )

    h, mu, logvar = latent(node_embeddings, batch_vector)

    # Only deterministic embedding should be returned
    assert isinstance(h, torch.Tensor)
    assert mu is None
    assert logvar is None

    # With a single pooling op producing a 32-dim vector → h must be 1×32
    assert h.shape == (2, 32)

# -------------------------------------------------------------------------
# Forward tests
# -------------------------------------------------------------------------

def test_latent_forward_multilayer(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=bottleneck_type,
        bottleneck_args=bottleneck_args,
        bottleneck_kwargs=bottleneck_kwargs,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    z, mu, logvar = latent(node_embeddings, batch_vector)

    assert isinstance(z, torch.Tensor)
    assert isinstance(mu, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
    assert z.shape == mu.shape == logvar.shape
    assert z.shape[-1] == 16

def test_latent_sampling_stochastic(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=bottleneck_type,
        bottleneck_args=bottleneck_args,
        bottleneck_kwargs=bottleneck_kwargs,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    z1, _, _ = latent(node_embeddings, batch_vector)
    z2, _, _ = latent(node_embeddings, batch_vector)
    assert not torch.allclose(z1, z2)

def test_forward_without_batch_succeeds(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    node_embeddings
):
    # Build a simple non‑VAE latent module
    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        mu_head_type=None,
        mu_head_args=None,
        mu_head_kwargs=None,
        logvar_head_type=None,
        logvar_head_args=None,
        logvar_head_kwargs=None,
    )

    # Forward without a batch vector
    h, mu, logvar = latent(node_embeddings)

    # Must succeed and behave as a deterministic non‑VAE module
    assert isinstance(h, torch.Tensor)
    assert mu is None
    assert logvar is None

    # With mean+max pooling, aggregator should concatenate to 64‑dim
    # (32‑dim from each pooling op)
    assert h.shape == (1, 64)

def test_heterogeneous_batch_pooling_minimal(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    node_embeddings
):
    """
    Minimal unit test: heterogeneous batches must produce exactly one
    pooled embedding per graph. No VAE, no bottleneck.
    """

    # Construct heterogeneous batch: 3 nodes in graph 0, 7 in graph 1
    batch = torch.tensor([0, 0, 0,   1, 1, 1, 1, 1, 1, 1])
    assert (batch == 0).sum() == 3
    assert (batch == 1).sum() == 7

    # Build latent module in deterministic non-VAE mode
    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,                # mean + max pooling
        aggregate_pooling_type=aggregate_type,       # concat
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        mu_head_type=None,
        mu_head_args=None,
        mu_head_kwargs=None,
        logvar_head_type=None,
        logvar_head_args=None,
        logvar_head_kwargs=None,
    )

    # Forward pass
    h, mu, logvar = latent(node_embeddings, batch)

    # Non-VAE mode → mu/logvar must be None
    assert mu is None
    assert logvar is None

    # Two graphs → two pooled embeddings
    assert h.shape[0] == 2

    # pooled_dim = 32(mean) + 32(max) = 64
    assert h.shape[1] == 64

    # Output must be finite and real-valued
    assert torch.isfinite(h).all()

# -------------------------------------------------------------------------
# Equivalence test: from_config vs direct construction
# -------------------------------------------------------------------------

def test_latent_from_config_equivalence(
    latent_config,
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    latent_cfg = GraphEmbeddingToLatent.from_config(latent_config)

    latent_direct = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=bottleneck_type,
        bottleneck_args=bottleneck_args,
        bottleneck_kwargs=bottleneck_kwargs,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    # synchronize parameters before comparison 
    # this working already shows that the architectures are equal
    latent_direct.load_state_dict(latent_cfg.state_dict())


    # Forward outputs should match (deterministically for mu/logvar; z is stochastic)
    _, mu_cfg, log_cfg = latent_cfg(node_embeddings, batch_vector)
    _, mu_dir, log_dir = latent_direct(node_embeddings, batch_vector)

    assert torch.allclose(mu_cfg, mu_dir)
    assert torch.allclose(log_cfg, log_dir)

    # Parameter names and shapes should match
    params_cfg = {k: v.shape for k, v in latent_cfg.named_parameters()}
    params_dir = {k: v.shape for k, v in latent_direct.named_parameters()}
    assert params_cfg == params_dir

# -------------------------------------------------------------------------
# Config validation
# -------------------------------------------------------------------------

def test_missing_pooling_layers_raises():
    cfg = {
        "aggregate_pooling_type": lambda seq: torch.cat(seq, dim=1),
        "aggregate_pooling_args": [],
        "aggregate_pooling_kwargs": {},
    }

    # Missing pooling_layers -> schema validation should fail
    with pytest.raises(jsonschema.ValidationError):
        GraphEmbeddingToLatent.from_config(cfg)

def test_missing_aggregate_pooling_type_raises(pooling_specs):
    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_args": [],
        "aggregate_pooling_kwargs": {},
    }

    # Missing aggregate_pooling_type → schema validation should fail
    with pytest.raises(jsonschema.ValidationError):
        GraphEmbeddingToLatent.from_config(cfg)

def test_mu_without_logvar_raises(pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
                                  mu_head_type, mu_head_args, mu_head_kwargs):
    # Only mu_head is provided → must raise ValueError during construction
    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "mu_head_type": mu_head_type,
        "mu_head_args": mu_head_args,
        "mu_head_kwargs": mu_head_kwargs,
    }

    with pytest.raises(ValueError):
        GraphEmbeddingToLatent.from_config(cfg)


def test_logvar_without_mu_raises(pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
                                  logvar_head_type, logvar_head_args, logvar_head_kwargs):
    # Only logvar_head is provided → must raise ValueError during construction
    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "logvar_head_type": logvar_head_type,
        "logvar_head_args": logvar_head_args,
        "logvar_head_kwargs": logvar_head_kwargs,
    }

    with pytest.raises(ValueError):
        GraphEmbeddingToLatent.from_config(cfg)

def test_malformed_pooling_entry_raises(aggregate_type, aggregate_args, aggregate_kwargs):
    # Pooling layers must be list of [type, args, kwargs]
    # Here we provide an invalid entry "not-a-triplet"
    cfg = {
        "pooling_layers": ["not-a-triplet"],
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
    }

    # Should fail JSON schema validation
    with pytest.raises(jsonschema.ValidationError):
        GraphEmbeddingToLatent.from_config(cfg)

# -------------------------------------------------------------------------
# No bottleneck in VAE mode (bottleneck_type=None)
# -------------------------------------------------------------------------

def test_no_bottleneck_vae_mode(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    mu_head_type, mu_head_args_no_bottle, mu_head_kwargs,
    logvar_head_type, logvar_head_args_no_bottle, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    # No bottleneck: pooled embedding (32) is fed directly into VAE heads.

    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "bottleneck_type": None,
        "bottleneck_args": None,
        "bottleneck_kwargs": None,
        "mu_head_type": mu_head_type,
        "mu_head_args": mu_head_args_no_bottle,
        "mu_head_kwargs": mu_head_kwargs,
        "logvar_head_type": logvar_head_type,
        "logvar_head_args": logvar_head_args_no_bottle,
        "logvar_head_kwargs": logvar_head_kwargs,
    }

    latent = GraphEmbeddingToLatent.from_config(cfg)

    z, mu, logvar = latent(node_embeddings, batch_vector)

    # All outputs must exist (VAE mode)
    assert isinstance(z, torch.Tensor)
    assert isinstance(mu, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)

    # Output dimension must match the heads' last layer (16)
    assert z.shape[-1] == 16
    assert mu.shape[-1] == 16
    assert logvar.shape[-1] == 16

# -------------------------------------------------------------------------
# No VAE heads → deterministic non‑VAE mode
# -------------------------------------------------------------------------

def test_no_vae_heads_nonvae_mode(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    node_embeddings, batch_vector
):
    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "mu_head_type": None,
        "mu_head_args": None,
        "mu_head_kwargs": None,
        "logvar_head_type": None,
        "logvar_head_args": None,
        "logvar_head_kwargs": None,
    }

    latent = GraphEmbeddingToLatent.from_config(cfg)

    # Forward twice: should be deterministic since no sampling occurs
    h1, mu1, log1 = latent(node_embeddings, batch_vector)
    h2, mu2, log2 = latent(node_embeddings, batch_vector)

    assert torch.allclose(h1, h2), "Non‑VAE latent must be deterministic"
    assert mu1 is None
    assert mu2 is None
    assert log1 is None
    assert log2 is None

# -------------------------------------------------------------------------
# Backprop tests
# -------------------------------------------------------------------------


def test_latent_gradient_flow_with_bottleneck(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    """Gradient must flow through pooling → aggregate → bottleneck → μ/logvar → z."""

    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=bottleneck_type,
        bottleneck_args=bottleneck_args,
        bottleneck_kwargs=bottleneck_kwargs,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    z, mu, logvar = latent(node_embeddings, batch_vector)
    loss = (z**2).sum() + (mu**2).sum() + (logvar**2).sum()
    loss.backward()

    for _, p in latent.named_parameters():
        assert p.grad is not None, "All parameters must receive gradients in full VAE mode."

def test_latent_gradient_flow_without_bottleneck(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    mu_head_type, mu_head_args_no_bottle, mu_head_kwargs,
    logvar_head_type, logvar_head_args_no_bottle, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    """Gradient must also flow when bottleneck is disabled (legal VAE configuration)."""

    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
        bottleneck_type=None,
        bottleneck_args=None,
        bottleneck_kwargs=None,
        mu_head_type=mu_head_type,
        mu_head_args=mu_head_args_no_bottle,
        mu_head_kwargs=mu_head_kwargs,
        logvar_head_type=logvar_head_type,
        logvar_head_args=logvar_head_args_no_bottle,
        logvar_head_kwargs=logvar_head_kwargs,
    )

    z, mu, logvar = latent(node_embeddings, batch_vector)
    loss = (z**2).sum() + (mu**2).sum() + (logvar**2).sum()
    loss.backward()

    for _, p in latent.named_parameters():
        assert p.grad is not None, "Parameters must receive gradients in no-bottleneck VAE mode."