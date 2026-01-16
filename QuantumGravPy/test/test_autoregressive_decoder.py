import QuantumGrav as QG
import pytest

import jsonschema

import torch


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# GNN-based decoder fixtures (global defaults)
# -------------------------------------------------------------------------

@pytest.fixture
def gru_args():
    # GRUCell(input_dim=32, hidden_dim=32)
    return [32, 32]

@pytest.fixture
def gru_kwargs():
    return {
        "bias": True
    }

@pytest.fixture
def parent_logit_mlp_type():
    return QG.models.LinearSequential

@pytest.fixture
def parent_logit_mlp_args():
    # dims, activations
    return [
        [[32 + 32, 1]],           # dims list
        [torch.nn.Identity],  # activations list
    ]

@pytest.fixture
def parent_logit_mlp_kwargs():
    return {
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }

@pytest.fixture
def decoder_init_type():
    return QG.models.LinearSequential

@pytest.fixture
def decoder_init_args():
    return [
        [[32, 32]],          # dims list: latent_dim=32 → hidden_dim=32
        [torch.nn.ReLU],     # activations list
    ]

@pytest.fixture
def decoder_init_kwargs():
    return {
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }

@pytest.fixture
def node_feature_decoder_type():
    return QG.models.LinearSequential

@pytest.fixture
def node_feature_decoder_args():
    return [
        [[32, 4]],            # dims list
        [torch.nn.ReLU],     # activations list
    ]

@pytest.fixture
def node_feature_decoder_kwargs():
    return {
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


@pytest.fixture
def bad_gru_args():
    # gru_args must be a list; here we deliberately provide an invalid type
    return "this_is_not_a_valid_arg_list"


@pytest.fixture
def deep_parent_logit_mlp_type():
    return QG.models.LinearSequential

@pytest.fixture
def deep_parent_logit_mlp_args():
    # dims, activations
    return [
        [
            [64, 64],
            [64, 32],
            [32, 1],
        ],
        [
            torch.nn.ReLU,
            torch.nn.ReLU,
            torch.nn.Identity,
        ],
    ]

@pytest.fixture
def deep_parent_logit_mlp_kwargs():
    return {
        "linear_kwargs": [{"bias": True}, {"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}, {}],
    }


# -------------------------------------------------------------------------
# Realistic decoder_init (latent → hidden state)
# -------------------------------------------------------------------------

@pytest.fixture
def deep_decoder_init_type():
    return QG.models.LinearSequential

@pytest.fixture
def deep_decoder_init_args():
    return [
        [
            [32, 64],
            [64, 32],
        ],
        [
            torch.nn.ReLU,
            torch.nn.ReLU,
        ],
    ]

@pytest.fixture
def deep_decoder_init_kwargs():
    return {
        "linear_kwargs": [{"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}],
    }


# -------------------------------------------------------------------------
# Realistic node_feature_decoder
# -------------------------------------------------------------------------

@pytest.fixture
def deep_node_feature_decoder_type():
    return QG.models.LinearSequential

@pytest.fixture
def deep_node_feature_decoder_args():
    return [
        [
            [32, 32],
            [32, 8],
        ],
        [
            torch.nn.ReLU,
            torch.nn.ReLU,
        ],
    ]

@pytest.fixture
def deep_node_feature_decoder_kwargs():
    return {
        "linear_kwargs": [{"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}],
    }

@pytest.fixture
def full_decoder_config(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    return {
        "gru_args": gru_args,
        "gru_kwargs": gru_kwargs,
        "parent_logit_mlp_type": parent_logit_mlp_type,
        "parent_logit_mlp_args": parent_logit_mlp_args,
        "parent_logit_mlp_kwargs": parent_logit_mlp_kwargs,
        "decoder_init_type": decoder_init_type,
        "decoder_init_args": decoder_init_args,
        "decoder_init_kwargs": decoder_init_kwargs,
        "node_feature_decoder_type": node_feature_decoder_type,
        "node_feature_decoder_args": node_feature_decoder_args,
        "node_feature_decoder_kwargs": node_feature_decoder_kwargs,
        "ancestor_suppression_strength": 0.7,
    }

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Construction of minimal model
# -------------------------------------------------------------------------

def test_minimal_construction(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    """
    Minimal valid decoder configuration:
    - a trivial decoder backbone (LinearSequential)
    - a trivial parent_logit_mlp (producing a single logit)
    This verifies that the class can be instantiated without optional components.
    """
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    assert isinstance(dec, QG.models.AutoregressiveDecoder)
    assert isinstance(dec.node_updater, torch.nn.Module)
    assert isinstance(dec.parent_logit_mlp, torch.nn.Module)
    assert dec.node_feature_decoder is None
    assert dec.ancestor_suppression_strength == 1.0
    assert dec.node_updater.hidden_dim == 32

# -------------------------------------------------------------------------
# decoder_init error tests
# -------------------------------------------------------------------------

def test_decoder_init_internal_exception_wrapped(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    """
    If decoder_init.forward raises an exception,
    the decoder must wrap it in ValueError with informative message.
    """

    class ExplodingInit(torch.nn.Module):
        def forward(self, z):
            raise RuntimeError("boom")

    with pytest.raises(ValueError) as excinfo:
        QG.models.AutoregressiveDecoder(
            gru_args=gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=parent_logit_mlp_type,
            parent_logit_mlp_args=parent_logit_mlp_args,
            parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
            decoder_init_type=ExplodingInit,
            decoder_init_args=[],
            decoder_init_kwargs={},
        )

    assert "decoder_init failed" in str(excinfo.value)
    assert "boom" in str(excinfo.value)

# -------------------------------------------------------------------------
# from_config equivalence with standard constructor
# -------------------------------------------------------------------------


def test_from_config_equivalence_full(
    full_decoder_config,
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    # Direct construction
    direct = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        decoder_init_type=decoder_init_type,
        decoder_init_args=decoder_init_args,
        decoder_init_kwargs=decoder_init_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
        ancestor_suppression_strength=0.7,
    )

    # Config construction
    cfg_model = QG.models.AutoregressiveDecoder.from_config(full_decoder_config)

    # Assert hidden_dim is correct for both models
    assert direct.node_updater.hidden_dim == 32
    assert cfg_model.node_updater.hidden_dim == 32

    # Compare parameter structures
    direct_params = {k: v.shape for k, v in direct.named_parameters()}
    cfg_params = {k: v.shape for k, v in cfg_model.named_parameters()}

    assert direct_params == cfg_params

    # ancestor_suppression_strength equality
    assert direct.ancestor_suppression_strength == cfg_model.ancestor_suppression_strength

    # Types of components must match
    assert type(direct.node_updater) == type(cfg_model.node_updater)
    assert type(direct.parent_logit_mlp) == type(cfg_model.parent_logit_mlp)
    assert type(direct.decoder_init) == type(cfg_model.decoder_init)
    assert type(direct.node_feature_decoder) == type(cfg_model.node_feature_decoder)

    # Forward pass outputs must have matching shapes
    dummy_latent = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32).unsqueeze(0)

    direct.train()
    cfg_model.train()
    direct_out = direct(dummy_latent, teacher_forcing_targets=teacher_forcing_targets)
    cfg_out = cfg_model(dummy_latent, teacher_forcing_targets=teacher_forcing_targets)
    
    # Both outputs are lists of length 1
    assert isinstance(direct_out, list)
    assert isinstance(cfg_out, list)
    assert len(direct_out) == 1
    assert len(cfg_out) == 1
    dL, dX, dlogp = direct_out[0]
    cL, cX, clogp = cfg_out[0]
    assert dL.shape == cL.shape
    if dX is not None and cX is not None:
        assert dX.shape == cX.shape
    assert (dX is None) == (cX is None)
    if dlogp is not None and clogp is not None:
        assert dlogp.shape == clogp.shape
    assert (dlogp is None) == (clogp is None)
    

# -------------------------------------------------------------------------
# Config validation: missing required keys
# -------------------------------------------------------------------------

def test_missing_gru_args_raises(
    parent_logit_mlp_type,
    parent_logit_mlp_args,
    parent_logit_mlp_kwargs,
):
    cfg = {
        # gru_args missing
        "parent_logit_mlp_type": parent_logit_mlp_type,
        "parent_logit_mlp_args": parent_logit_mlp_args,
        "parent_logit_mlp_kwargs": parent_logit_mlp_kwargs,
    }

    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder.from_config(cfg)


def test_missing_parent_logit_mlp_type_raises(
    gru_args, gru_kwargs
):
    cfg = {
        "gru_args": gru_args,
        "gru_kwargs": gru_kwargs,
        # missing "parent_logit_mlp_type"
    }

    with pytest.raises(jsonschema.ValidationError):
        QG.models.AutoregressiveDecoder.from_config(cfg)
        
# -------------------------------------------------------------------------
# malformed decoder args or kwargs must raise schema validation error
# -------------------------------------------------------------------------

def test_malformed_decoder_args_raises(
    bad_gru_args,
    gru_kwargs,
    parent_logit_mlp_type,
    parent_logit_mlp_args,
    parent_logit_mlp_kwargs,
):
    # invalid config: gru_args must be a list, not a string
    cfg = {
        "gru_args": bad_gru_args,       # malformed
        "gru_kwargs": gru_kwargs,
        "parent_logit_mlp_type": parent_logit_mlp_type,
        "parent_logit_mlp_args": parent_logit_mlp_args,
        "parent_logit_mlp_kwargs": parent_logit_mlp_kwargs,
    }

    with pytest.raises(jsonschema.ValidationError):
        QG.models.AutoregressiveDecoder.from_config(cfg)

 # -------------------------------------------------------------------------
 # training mode requires teacher_forcing_targets
 # -------------------------------------------------------------------------

def test_training_requires_teacher_forcing(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    """
    The decoder must raise ValueError if called in training mode without
    teacher_forcing_targets.
    """
    latent_z = torch.randn(1, 32)
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()  # training mode enforces teacher forcing

    with pytest.raises(ValueError):
        dec(latent_z, teacher_forcing_targets=None)
 
# -------------------------------------------------------------------------
# forward pass with teacher forcing -- training mode
# -------------------------------------------------------------------------

def test_forward_training_with_teacher_forcing_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_vector_tf = torch.randn(32)

    teacher_forcing_targets_full = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()
    L, X, logp = dec(latent_vector_tf, teacher_forcing_targets=teacher_forcing_targets_full)

    assert L.shape[0] == teacher_forcing_targets_full.shape[0]
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is not None

def test_forward_training_with_teacher_forcing_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_vector_tf = torch.randn(2, 32)

    teacher_forcing_targets_full = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.float32).unsqueeze(0)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()
    L, X, logp = dec(latent_vector_tf, teacher_forcing_targets=teacher_forcing_targets_full)[0]

    assert L.shape[0] == teacher_forcing_targets_full.shape[1]
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is not None

# -------------------------------------------------------------------------
# forward pass with teacher forcing -- evaluation mode
# -------------------------------------------------------------------------

def test_forward_eval_with_teacher_forcing_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_vector_tf = torch.randn(32)

    teacher_forcing_targets_full = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()
    L, X, logp = dec(latent_vector_tf, teacher_forcing_targets=teacher_forcing_targets_full)

    assert L.shape[0] == teacher_forcing_targets_full.shape[0]
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is not None

def test_forward_eval_with_teacher_forcing_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_vector_tf = torch.randn(2, 32)

    teacher_forcing_targets_full = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.float32).unsqueeze(0)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()
    L, X, logp = dec(latent_vector_tf, teacher_forcing_targets=teacher_forcing_targets_full)[0]

    assert L.shape[0] == teacher_forcing_targets_full.shape[1]
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is not None

# -------------------------------------------------------------------------
# forward pass without teacher forcing (& evaluation mode)
# -------------------------------------------------------------------------

def test_eval_autoregressive_sampling_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_eval_z = torch.randn(32)
    eval_atom_count = 5
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    L, X, logp = dec(latent_eval_z, atom_count=eval_atom_count)

    assert L.shape == (eval_atom_count, eval_atom_count)
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is None

def test_eval_autoregressive_sampling_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_eval_z = torch.randn(2, 32)
    eval_atom_count = 5
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    L, X, logp = dec(latent_eval_z, atom_count=eval_atom_count)[0]

    assert L.shape == (eval_atom_count, eval_atom_count)
    assert torch.allclose(L, torch.triu(L))
    assert X is None
    assert logp is None

# -------------------------------------------------------------------------
# forward: eval mode without teacher_forcing requires atom_count
# -------------------------------------------------------------------------

def test_eval_requires_atom_count_when_no_teacher_forcing(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(1, 32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    # No teacher forcing and no atom_count → must raise
    import pytest
    with pytest.raises(ValueError):
        dec(z, teacher_forcing_targets=None, atom_count=None)

# -------------------------------------------------------------------------
# forward: atom_count must be ignored when teacher_forcing_targets is provided
# -------------------------------------------------------------------------

def test_atom_count_ignored_when_teacher_forcing_given(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(32)

    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()

    # Provide conflicting atom_count (should be ignored)
    atom_count_conflict = 10

    L, X, logp = dec(z, teacher_forcing_targets=teacher_forcing_targets, atom_count=atom_count_conflict)

    # teacher_forcing_targets determine N_max, so shape must match teacher_forcing_targets
    assert L.shape[0] == teacher_forcing_targets.shape[0]
    assert L.shape[1] == teacher_forcing_targets.shape[0]
    assert torch.allclose(L, torch.triu(L))

    # logp must be computed in training mode
    assert logp is not None

    # node features are disabled here
    assert X is None

# -------------------------------------------------------------------------
# node_feature_decoder forward pass -- training mode with teacher forcing
# -------------------------------------------------------------------------

def test_node_feature_decoder_forward_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32).unsqueeze(0)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
    )
    dec.train()

    L, X, logp = dec(z, teacher_forcing_targets=teacher_forcing_targets)[0]

    # Node feature decoder must produce feature matrix
    assert torch.allclose(L, torch.triu(L))
    assert X is not None
    assert X.shape[0] == teacher_forcing_targets.shape[1]
    # Must have positive feature dimension
    assert X.shape[1] == 4  # as per node_feature_decoder_args
    # log-likelihood must still be computed
    assert logp is not None

def test_node_feature_decoder_forward_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
    )
    dec.train()

    L, X, logp = dec(z, teacher_forcing_targets=teacher_forcing_targets)

    # Node feature decoder must produce feature matrix
    assert torch.allclose(L, torch.triu(L))
    assert X is not None
    assert X.shape[0] == teacher_forcing_targets.shape[0]
    # Must have positive feature dimension
    assert X.shape[1] == 4  # as per node_feature_decoder_args
    # log-likelihood must still be computed
    assert logp is not None

# -------------------------------------------------------------------------
# Full construction and forward pass with realistic deep nets and all optional components
# -------------------------------------------------------------------------

def test_full_construction_with_deep_components(
    gru_args, gru_kwargs,
    deep_parent_logit_mlp_type, deep_parent_logit_mlp_args, deep_parent_logit_mlp_kwargs,
    deep_decoder_init_type, deep_decoder_init_args, deep_decoder_init_kwargs,
    deep_node_feature_decoder_type, deep_node_feature_decoder_args, deep_node_feature_decoder_kwargs,
):
    latent = torch.randn(1, 32)
    teacher = torch.tensor([
        [0,0,0],
        [1,0,0],
        [0,1,0]
    ], dtype=torch.float32).unsqueeze(0)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,

        parent_logit_mlp_type=deep_parent_logit_mlp_type,
        parent_logit_mlp_args=deep_parent_logit_mlp_args,
        parent_logit_mlp_kwargs=deep_parent_logit_mlp_kwargs,

        decoder_init_type=deep_decoder_init_type,
        decoder_init_args=deep_decoder_init_args,
        decoder_init_kwargs=deep_decoder_init_kwargs,

        node_feature_decoder_type=deep_node_feature_decoder_type,
        node_feature_decoder_args=deep_node_feature_decoder_args,
        node_feature_decoder_kwargs=deep_node_feature_decoder_kwargs,

        ancestor_suppression_strength=0.7,
    )

    # Structural checks
    assert isinstance(dec, QG.models.AutoregressiveDecoder)
    assert isinstance(dec.node_updater, torch.nn.Module)
    assert isinstance(dec.parent_logit_mlp, torch.nn.Module)
    assert isinstance(dec.decoder_init, torch.nn.Module)
    assert isinstance(dec.node_feature_decoder, torch.nn.Module)
    assert dec.node_updater.hidden_dim == 32

    # Forward behaviour
    dec.train()
    out = dec(latent, teacher_forcing_targets=teacher)
    assert isinstance(out, list)
    assert len(out) == 1

    L, X, logp = out[0]
    assert L.shape == teacher.shape[1:]
    assert torch.allclose(L, torch.triu(L))
    assert X.shape[0] == teacher.shape[1]
    assert X.shape[1] == 8   # from deep_node_feature_decoder_args
    assert logp is not None

# -------------------------------------------------------------------------
# Backward pass gradient flow through full decoder with deep components
# -------------------------------------------------------------------------


def test_full_decoder_gradient_flow_deep_components(
    gru_args, gru_kwargs,
    deep_parent_logit_mlp_type, deep_parent_logit_mlp_args, deep_parent_logit_mlp_kwargs,
    deep_decoder_init_type, deep_decoder_init_args, deep_decoder_init_kwargs,
    deep_node_feature_decoder_type, deep_node_feature_decoder_args, deep_node_feature_decoder_kwargs,
):
    """
    Ensure that the FULL decoder (deep parent MLP, deep init, deep node-feature decoder)
    has proper end-to-end gradient flow. All parameters must receive a non-None gradient.
    """

    # Build full model
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,

        parent_logit_mlp_type=deep_parent_logit_mlp_type,
        parent_logit_mlp_args=deep_parent_logit_mlp_args,
        parent_logit_mlp_kwargs=deep_parent_logit_mlp_kwargs,

        decoder_init_type=deep_decoder_init_type,
        decoder_init_args=deep_decoder_init_args,
        decoder_init_kwargs=deep_decoder_init_kwargs,

        node_feature_decoder_type=deep_node_feature_decoder_type,
        node_feature_decoder_args=deep_node_feature_decoder_args,
        node_feature_decoder_kwargs=deep_node_feature_decoder_kwargs,

        ancestor_suppression_strength=0.7,
    )

    dec.train()

    # Latent batch
    z = torch.randn(1, 32, requires_grad=True)

    # Teacher forcing batch (must match N × N)
    teacher = torch.tensor(
        [[[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0]]],
        dtype=torch.float32,
    )

    # Forward
    out = dec(z, teacher_forcing_targets=teacher)
    assert isinstance(out, list) and len(out) == 1
    L, X, logp = out[0]
    assert torch.allclose(L, torch.triu(L))

    # Scalar loss that pulls on ALL modules
    # Combine link reconstruction + node features + likelihood
    loss = L.sum()
    if X is not None:
        loss = loss + X.sum()
    if logp is not None:
        loss = loss - logp.sum()

    loss.backward()

    # Check every parameter receives gradient
    for name, p in dec.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradient"

# -------------------------------------------------------------------------
# reconstruct_link_matrix
# -------------------------------------------------------------------------

def test_reconstruct_link_matrix_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(2, 32)
    atom_count = 4

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )
    dec.eval()

    L = dec.reconstruct_link_matrix(z, atom_count)[0]

    assert isinstance(L, torch.Tensor)
    assert L.shape == (atom_count, atom_count)
    assert L.dtype == torch.float32
    assert torch.allclose(L, torch.triu(L))
    assert torch.all((L == 0) | (L == 1))


def test_reconstruct_link_matrix_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
):
    z = torch.randn(32)
    atom_count = 4

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )
    dec.eval()

    L = dec.reconstruct_link_matrix(z, atom_count)

    assert isinstance(L, torch.Tensor)
    assert L.shape == (atom_count, atom_count)
    assert L.dtype == torch.float32
    assert torch.allclose(L, torch.triu(L))
    assert torch.all((L == 0) | (L == 1))

# -------------------------------------------------------------------------
# reconstruct_node_features
# -------------------------------------------------------------------------

def test_reconstruct_node_features_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(1, 32)
    atom_count = 4

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
    )
    dec.eval()

    L, X = dec.reconstruct_node_features(z, atom_count)[0]

    # L must be a square adjacency matrix
    assert L.shape == (atom_count, atom_count)
    assert torch.all((L == 0) | (L == 1))

    # X must exist because node_feature_decoder is configured
    assert X is not None
    assert X.shape[0] == atom_count
    assert X.shape[1] == 4 

def test_reconstruct_node_features_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(32)
    atom_count = 4

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
    )
    dec.eval()

    L, X = dec.reconstruct_node_features(z, atom_count)

    # L must be a square adjacency matrix
    assert L.shape == (atom_count, atom_count)
    assert torch.all((L == 0) | (L == 1))

    # X must exist because node_feature_decoder is configured
    assert X is not None
    assert X.shape[0] == atom_count
    assert X.shape[1] == 4 

# -------------------------------------------------------------------------
# reconstruct_node_features must raise without node_feature_decoder
# -------------------------------------------------------------------------

def test_reconstruct_node_features_without_decoder_raises(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(1, 32)
    atom_count = 4

    # Construct decoder WITHOUT node_feature_decoder
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    # Must raise because node_feature_decoder is None
    with pytest.raises(RuntimeError):
        dec.reconstruct_node_features(z, atom_count)

# -------------------------------------------------------------------------
# compute_normalized_log_likelihood
# -------------------------------------------------------------------------

def test_compute_normalized_log_likelihood_single(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(32)

    teacher = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()

    n = teacher.size(0)
    norm = n * (n - 1) / 2

    ll = dec.compute_normalized_log_likelihood(z, teacher)

    assert isinstance(ll, torch.Tensor)
    assert ll.numel() == 1
    # Normalized log-likelihood must be finite
    assert torch.isfinite(ll)
    # Check magnitude roughly reasonable: abs(ll * norm) = unnormalized ll
    assert torch.isfinite(ll * norm)


def test_compute_normalized_log_likelihood_batched(
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(2, 32)

    teacher_batch = torch.tensor([
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()

    ll_batch = dec.compute_normalized_log_likelihood(z, teacher_batch)

    assert isinstance(ll_batch, torch.Tensor)
    assert ll_batch.shape == (2,)
    assert torch.isfinite(ll_batch).all()

# -------------------------------------------------------------------------
# save/load equivalence test
# -------------------------------------------------------------------------

def test_save_and_load_equivalence(
    tmp_path,
    full_decoder_config,
    gru_args, gru_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    # Build model
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        decoder_init_type=decoder_init_type,
        decoder_init_args=decoder_init_args,
        decoder_init_kwargs=decoder_init_kwargs,
        node_feature_decoder_type=node_feature_decoder_type,
        node_feature_decoder_args=node_feature_decoder_args,
        node_feature_decoder_kwargs=node_feature_decoder_kwargs,
        ancestor_suppression_strength=full_decoder_config["ancestor_suppression_strength"],
    )

    # Save
    save_path = tmp_path / "decoder_test_weights.pt"
    dec.save(str(save_path))

    # Load
    dec2 = QG.models.AutoregressiveDecoder.load(str(save_path), full_decoder_config)

    # Compare parameter shapes and values
    for (n1, p1), (n2, p2) in zip(dec.named_parameters(), dec2.named_parameters()):
        assert n1 == n2
        assert p1.shape == p2.shape
        assert torch.allclose(p1, p2)


def test_named_parameters_include_expected_modules(
    gru_args, gru_kwargs,
    deep_parent_logit_mlp_type, deep_parent_logit_mlp_args, deep_parent_logit_mlp_kwargs,
    deep_decoder_init_type, deep_decoder_init_args, deep_decoder_init_kwargs,
    deep_node_feature_decoder_type, deep_node_feature_decoder_args, deep_node_feature_decoder_kwargs,
):
    dec = QG.models.AutoregressiveDecoder(
        gru_args=gru_args,
        gru_kwargs=gru_kwargs,
        parent_logit_mlp_type=deep_parent_logit_mlp_type,
        parent_logit_mlp_args=deep_parent_logit_mlp_args,
        parent_logit_mlp_kwargs=deep_parent_logit_mlp_kwargs,
        decoder_init_type=deep_decoder_init_type,
        decoder_init_args=deep_decoder_init_args,
        decoder_init_kwargs=deep_decoder_init_kwargs,
        node_feature_decoder_type=deep_node_feature_decoder_type,
        node_feature_decoder_args=deep_node_feature_decoder_args,
        node_feature_decoder_kwargs=deep_node_feature_decoder_kwargs,
    )

    param_names = [name for name, _ in dec.named_parameters()]
    for module_name in ("node_updater", "parent_logit_mlp", "decoder_init", "node_feature_decoder"):
        module = getattr(dec, module_name)
        assert module is not None
        assert any(name.startswith(f"{module_name}.") for name in param_names)
