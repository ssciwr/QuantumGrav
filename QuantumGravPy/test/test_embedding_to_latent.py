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
    # 32 → 64 → 16
    return [[32, 64, 16]]


@pytest.fixture
def bottleneck_kwargs():
    return {
        "activation": torch.nn.ReLU,
        "dropout": 0.1,
        "linear_kwargs": [{"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}],
    }


@pytest.fixture
def mu_head_type():
    return QG.models.LinearSequential


@pytest.fixture
def mu_head_args():
    # 16 → 16
    return [[16, 16]]


@pytest.fixture
def mu_head_kwargs():
    return {
        "activation": torch.nn.Identity,
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


@pytest.fixture
def logvar_head_type():
    return QG.models.LinearSequential


@pytest.fixture
def logvar_head_args():
    # 16 → 16
    return [[16, 16]]


@pytest.fixture
def logvar_head_kwargs():
    return {
        "activation": torch.nn.Identity,
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


def test_latent_gradient_flow(
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
    loss = (z**2).sum() + (mu**2).sum() + (logvar**2).sum()
    loss.backward()

    for name, p in latent.named_parameters():
        assert p.grad is not None


def test_latent_state_dict_roundtrip(
    tmp_path,
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    bottleneck_type, bottleneck_args, bottleneck_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    latent1 = GraphEmbeddingToLatent(
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

    torch.save(latent1.state_dict(), tmp_path / "lat.pt")

    latent2 = GraphEmbeddingToLatent(
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
    latent2.load_state_dict(torch.load(tmp_path / "lat.pt"))

    _, mu1, log1 = latent1(node_embeddings, batch_vector)
    _, mu2, log2 = latent2(node_embeddings, batch_vector)

    assert torch.allclose(mu1, mu2)
    assert torch.allclose(log1, log2)

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
# Config validation: missing pooling_layers
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

# -------------------------------------------------------------------------
# Config validation: missing aggregate_pooling_type
# -------------------------------------------------------------------------

def test_missing_aggregate_pooling_type_raises(pooling_specs):
    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_args": [],
        "aggregate_pooling_kwargs": {},
    }

    # Missing aggregate_pooling_type → schema validation should fail
    with pytest.raises(jsonschema.ValidationError):
        GraphEmbeddingToLatent.from_config(cfg)

# -------------------------------------------------------------------------
# Config validation: mu_head without logvar_head (and vice versa)
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Config validation: malformed pooling entry
# -------------------------------------------------------------------------

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
# Dimensional consistency: bottleneck dim mismatch must raise
# -------------------------------------------------------------------------

def test_bottleneck_dim_mismatch_raises(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
    node_embeddings, batch_vector
):
    # Construct a bottleneck expecting WRONG input dim: 64 instead of 32
    mismatched_bottleneck_type = QG.models.LinearSequential
    mismatched_bottleneck_args = [[64, 32, 16]]  # first layer expects 64-dim input

    cfg = {
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": aggregate_type,
        "aggregate_pooling_args": aggregate_args,
        "aggregate_pooling_kwargs": aggregate_kwargs,
        "bottleneck_type": mismatched_bottleneck_type,
        "bottleneck_args": mismatched_bottleneck_args,
        "bottleneck_kwargs": {},
        "mu_head_type": mu_head_type,
        "mu_head_args": mu_head_args,
        "mu_head_kwargs": mu_head_kwargs,
        "logvar_head_type": logvar_head_type,
        "logvar_head_args": logvar_head_args,
        "logvar_head_kwargs": logvar_head_kwargs,
    }

    latent = GraphEmbeddingToLatent.from_config(cfg)

    # Forward should fail due to dimension mismatch
    import pytest
    with pytest.raises(RuntimeError):
        latent(node_embeddings, batch_vector)

# -------------------------------------------------------------------------
# Multiple pooling ops with different output dims must concatenate
# -------------------------------------------------------------------------

def test_multiple_pooling_layers_concat_correctly(
    aggregate_type, aggregate_args, aggregate_kwargs, node_embeddings
):
    # Define two custom pooling ops with different output dimensions
    class Pool32(torch.nn.Module):
        def forward(self, x, batch=None):
            return torch.randn(1, 32)

    class Pool48(torch.nn.Module):
        def forward(self, x, batch=None):
            return torch.randn(1, 48)

    pooling_specs = [
        [Pool32, [], {}],
        [Pool48, [], {}],
    ]

    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=aggregate_type,
        aggregate_pooling_args=aggregate_args,
        aggregate_pooling_kwargs=aggregate_kwargs,
    )

    # Forward (non-VAE mode → return (h, None, None))
    h, mu, logvar = latent(node_embeddings)

    assert h.shape == (1, 80)     # 32 + 48
    assert mu is None
    assert logvar is None

# -------------------------------------------------------------------------
# Faulty aggregator must raise an error
# -------------------------------------------------------------------------

def test_faulty_aggregator_raises(pooling_specs, node_embeddings, batch_vector):
    # Aggregator that returns an invalid object (list instead of tensor)
    def faulty_agg(seq):
        return seq   # should be a tensor, not a list

    latent = GraphEmbeddingToLatent(
        pooling_layers=pooling_specs,
        aggregate_pooling_type=faulty_agg,
        aggregate_pooling_args=[],
        aggregate_pooling_kwargs={},
    )

    import pytest
    # Forward must fail when subsequent layers expect a tensor
    with pytest.raises(Exception):
        latent(node_embeddings, batch_vector)

# -------------------------------------------------------------------------
# No bottleneck in VAE mode (bottleneck_type=None)
# -------------------------------------------------------------------------

def test_no_bottleneck_vae_mode(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    mu_head_type, mu_head_args, mu_head_kwargs,
    logvar_head_type, logvar_head_args, logvar_head_kwargs,
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
        "mu_head_args": mu_head_args,
        "mu_head_kwargs": mu_head_kwargs,
        "logvar_head_type": logvar_head_type,
        "logvar_head_args": logvar_head_args,
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
# Forward without batch must succeed (single‑graph mode)
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Non‑VAE forward must be deterministic
# -------------------------------------------------------------------------

def test_nonvae_forward_deterministic(
    pooling_specs, aggregate_type, aggregate_args, aggregate_kwargs,
    node_embeddings, batch_vector
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

    # Forward twice: must return identical outputs
    h1, mu1, log1 = latent(node_embeddings, batch_vector)
    h2, mu2, log2 = latent(node_embeddings, batch_vector)

    assert torch.allclose(h1, h2)
    assert mu1 is None and mu2 is None
    assert log1 is None and log2 is None