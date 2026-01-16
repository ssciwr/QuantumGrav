import QuantumGrav as QG
import pytest

import torch


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def gru_type():
    return torch.nn.GRUCell


@pytest.fixture
def input_dim():
    return 32


@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def gru_args(input_dim, hidden_dim):
    # Correct API: first two values = input_dim, hidden_dim
    return [input_dim, hidden_dim]


@pytest.fixture
def gru_kwargs():
    # minimal reproducible configuration
    return {"bias": True}


@pytest.fixture
def aggregation_method():
    # use a simple deterministic method
    return "mean"

@pytest.fixture
def config(gru_type, gru_args, gru_kwargs, aggregation_method):
    """
    A manually-written configuration dictionary for NodeUpdateGRU,
    matching the JSON schema and not relying on to_config().
    """
    return {
        "gru_type": gru_type,
        "gru_args": gru_args,
        "gru_kwargs": gru_kwargs,
        "aggregation_method": aggregation_method,
        "pooling_mlp_type": None,
        "pooling_mlp_args": [],
        "pooling_mlp_kwargs": {},
    }


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Instantiation tests
# -------------------------------------------------------------------------


def test_gru_construction_basic(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim, hidden_dim,
):
    """
    Basic construction test for NodeUpdateGRU (API-correct).

    Ensures:
    - the module instantiates without errors
    - GRUCell exists and has correct dimensions
    - input_dim and hidden_dim match gru_args
    - forward pass works for dummy parent states
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    # --- structural tests ---
    assert hasattr(gru, "gru"), "GRU cell missing."
    assert isinstance(gru.gru, torch.nn.GRUCell), "GRU must be GRUCell-based."

    assert gru.in_dim == input_dim
    assert gru.hidden_dim == hidden_dim

    # --- dummy parent states ---
    parent_states = torch.randn(5, input_dim)

    # Forward should not raise
    out = gru(parent_states)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (hidden_dim,)

def test_gru_mlp_aggregation_instantiation(gru_args, gru_kwargs):
    class DummyMLP(torch.nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.lin = torch.nn.Linear(in_dim, out_dim)

        def forward(self, x):
            # x: (num_parents, hidden_dim)
            return self.lin(x.mean(dim=0))

    gru = QG.models.NodeUpdateGRU(
        gru_type=torch.nn.GRUCell,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method="mlp",
        pooling_mlp_type=DummyMLP,
        pooling_mlp_args=[gru_args[1], gru_args[1]],   # hidden_dim → hidden_dim
        pooling_mlp_kwargs={},
    )

    assert isinstance(gru.pooling_mlp, DummyMLP)
        
def test_gru_args_non_integer(
    gru_type, gru_kwargs, aggregation_method,
):
    """
    NodeUpdateGRU requires that the first two elements of gru_args are
    integer input_dim and hidden_dim. Non‑integer values must raise ValueError.
    """
    bad_gru_args = ["not_an_int", 32]  # invalid input_dim

    with pytest.raises(TypeError):
        QG.models.NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            aggregation_method=aggregation_method,
        )

    bad_gru_args = [32, "not_an_int"]  # invalid hidden_dim

    with pytest.raises(TypeError):
        QG.models.NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            aggregation_method=aggregation_method,
        )


def test_gru_args_non_positive(
    gru_type, gru_kwargs, aggregation_method,
):
    """
    NodeUpdateGRU requires the first two gru_args to be strictly positive integers.
    Zero or negative values must raise ValueError.
    """

    # Case 1: input_dim <= 0
    bad_gru_args = [0, 32]
    with pytest.raises(ValueError):
        QG.models.NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            aggregation_method=aggregation_method,
        )

    # Case 2: hidden_dim <= 0
    bad_gru_args = [32, -1]
    with pytest.raises(ValueError):
        QG.models.NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            aggregation_method=aggregation_method,
        )

def test_invalid_gru_args_too_short(
        gru_type, gru_kwargs, aggregation_method,
):
    """
    gru_args must contain at least [input_dim, hidden_dim].
    """
    bad_args = [32]  # missing hidden_dim
    with pytest.raises(ValueError):
        QG.models.NodeUpdateGRU(
            gru_type=gru_type,
            gru_args=bad_args,
            gru_kwargs=gru_kwargs,
            aggregation_method=aggregation_method,
        )

def test_gru_mlp_requires_pooling_mlp_type(gru_args, gru_kwargs):
    with pytest.raises(ValueError):
        QG.models.NodeUpdateGRU(
            gru_type=torch.nn.GRUCell,
            gru_args=gru_args,
            gru_kwargs=gru_kwargs,
            aggregation_method="mlp",
            pooling_mlp_type=None,
        )

# -------------------------------------------------------------------------
# Forward tests
# -------------------------------------------------------------------------

def test_gru_forward_basic(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim, hidden_dim,
):
    """
    Verifies:
    - forward() accepts parent_states of shape (N, input_dim)
    - output is a tensor of shape (hidden_dim,)
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    # N parents, each with input_dim (== hidden_dim for GRUCell input)
    parent_states = torch.randn(7, input_dim)

    out = gru(parent_states)

    assert isinstance(out, torch.Tensor)
    # GRUCell returns (hidden_dim,)
    assert out.shape == (hidden_dim,)

def test_gru_forward_heterogeneous_parent_counts(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim, hidden_dim,
):
    """
    Tests that NodeUpdateGRU handles different numbers of parent states
    (heterogeneous parent sets), i.e. variable N in (N, input_dim).

    Ensures:
    - No shape assumptions about fixed N
    - Output always has shape (hidden_dim,)
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    # Example heterogeneous parent sets for 3 nodes:
    parent_sets = [
        torch.randn(1, input_dim),        # Node 1 has 1 parent
        torch.randn(5, input_dim),        # Node 2 has 5 parents
        torch.randn(12, input_dim),       # Node 3 has 12 parents
    ]

    for parents in parent_sets:
        out = gru(parents)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (hidden_dim,), (
            f"Expected output shape (hidden_dim,), got {out.shape}"
        )

def test_gru_single_parent_case(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim, hidden_dim,
):
    """
    Single-parent case: parent_states has shape (1, input_dim).
    Must return a valid hidden state of shape (hidden_dim,).
    """
    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    parent_states = torch.randn(1, input_dim)
    out = gru(parent_states)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (hidden_dim,)

def test_gru_mlp_forward_works(gru_args, gru_kwargs):
    class DummyMLP(torch.nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.lin = torch.nn.Linear(d_in, d_out)

        def forward(self, x):
            return self.lin(x.mean(dim=0))

    gru = QG.models.NodeUpdateGRU(
        gru_type=torch.nn.GRUCell,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method="mlp",
        pooling_mlp_type=DummyMLP,
        pooling_mlp_args=[gru_args[0], gru_args[0]],
        pooling_mlp_kwargs={},
    )

    parent_states = torch.randn(5, gru_args[0])
    out = gru(parent_states)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (gru_args[1],)

def test_gru_zero_parents_raises(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim,
 ):
    """
    A node with zero parents yields parent_states shape (0, input_dim).
    Current GRU design does not define a default aggregation rule for the empty
    set, and such cases must raise a clear error rather than silently produce NaNs.
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    parent_states = torch.randn(0, input_dim)  # empty parent set

    with pytest.raises(Exception):
        gru(parent_states)

def test_gru_parent_state_feature_dim_mismatch_raises(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim,
):
    """
    NodeUpdateGRU must raise a ValueError when parent_states has the wrong
    feature dimension (i.e. parent_states.shape[1] != input_dim).
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )

    # Wrong feature dimension: use input_dim + 1
    wrong_dim = input_dim + 1
    parent_states = torch.randn(4, wrong_dim)

    with pytest.raises(ValueError):
        gru(parent_states)

# -------------------------------------------------------------------------
# Backpropagation tests
# -------------------------------------------------------------------------


def test_gru_gradient_flow(
    gru_type, gru_args, gru_kwargs, aggregation_method,
    input_dim,
):
    """
    Ensure gradients flow from output back to GRU parameters.
    """

    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )
    gru.train()

    parent_states = torch.randn(10, input_dim, requires_grad=True)
    out = gru(parent_states)
    loss = out.sum()
    loss.backward()

    # check gradients
    assert parent_states.grad is not None
    for name, param in gru.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"

# -------------------------------------------------------------------------
# Instantiation from config tests
# -------------------------------------------------------------------------

def test_gru_from_config_equivalence(aggregation_method, gru_args, gru_kwargs):
    """
    Unified equivalence test for NodeUpdateGRU:
    - works for both standard aggregation ("mean") and MLP pooling ("mlp")
    - ensures config → model equality after loading state_dict
    - ensures inequality before state_dict transfer
    """

    input_dim = gru_args[0]
    hidden_dim = gru_args[0]

    # --- define DummyMLP only when needed ---
    class DummyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(input_dim, hidden_dim)

        def forward(self, x):
            # (N, input_dim) → (hidden_dim,)
            return self.lin(x.mean(dim=0))

    # --- Build configuration dict ---
    cfg = {
        "gru_type": torch.nn.GRUCell,
        "gru_args": gru_args,
        "gru_kwargs": gru_kwargs,
        "aggregation_method": aggregation_method,
        "pooling_mlp_type": DummyMLP,
        "pooling_mlp_args": [],
        "pooling_mlp_kwargs": {},
    }

    # --- Direct construction ---
    gru_direct = QG.models.NodeUpdateGRU(**cfg)

    # --- From-config construction ---
    gru_cfg = QG.models.NodeUpdateGRU.from_config(cfg)

    # --- Check architecture ---
    assert gru_cfg.in_dim == gru_direct.in_dim
    assert gru_cfg.hidden_dim == gru_direct.hidden_dim
    assert type(gru_cfg.gru) is type(gru_direct.gru)
    assert gru_cfg.aggregation_method == gru_direct.aggregation_method

    # --- Evaluate on some parent states ---
    parent_states = torch.randn(5, input_dim)

    out_direct = gru_direct(parent_states)
    out_cfg = gru_cfg(parent_states)

    # BEFORE loading the state_dict, outputs must differ
    assert not torch.allclose(out_direct, out_cfg), \
        "Unexpectedly identical outputs before transferring state_dict."

    # AFTER loading state_dict, outputs must match
    gru_cfg.load_state_dict(gru_direct.state_dict())
    out_cfg_after = gru_cfg(parent_states)

    assert torch.allclose(out_direct, out_cfg_after), \
        "Outputs do not match after state_dict transfer."
# -------------------------------------------------------------------------
# Save and load tests
# -------------------------------------------------------------------------

def test_gru_save_and_load(
    tmp_path,
    gru_type, gru_args, gru_kwargs, aggregation_method, input_dim
):
    """
    Test that NodeUpdateGRU.save() and NodeUpdateGRU.load() correctly
    persist and restore both configuration and parameters.
    """

    # ----- original model -----
    gru = QG.models.NodeUpdateGRU(
        gru_type=gru_type,
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        aggregation_method=aggregation_method,
    )
    gru.eval()

    parent_states = torch.randn(6, input_dim)
    out_before = gru(parent_states)

    # ----- save to file -----
    filepath = tmp_path / "gru_test.pt"
    gru.save(filepath)

    assert filepath.exists(), "GRU save() did not create a file."

    # ----- load from file -----
    gru_loaded = QG.models.NodeUpdateGRU.load(filepath)
    gru_loaded.eval()

    # Architecture checks
    assert gru_loaded.in_dim == gru.in_dim
    assert gru_loaded.hidden_dim == gru.hidden_dim
    assert gru_loaded.aggregation_method == gru.aggregation_method
    assert type(gru_loaded.gru) is type(gru.gru)

    # Parameter equivalence
    for (n1, p1), (n2, p2) in zip(
        gru.named_parameters(), gru_loaded.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Parameter mismatch after load: {n1} vs {n2}"

    # Forward equivalence
    out_after = gru_loaded(parent_states)
    assert torch.allclose(out_before, out_after), \
        "Loaded GRU does not reproduce outputs of the saved GRU."