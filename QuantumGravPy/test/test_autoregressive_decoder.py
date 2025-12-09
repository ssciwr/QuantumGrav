import QuantumGrav as QG
import pytest

import jsonschema

import torch
import torch_geometric


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# GNN-based decoder fixtures (global defaults)
# -------------------------------------------------------------------------

@pytest.fixture
def decoder_type():
    return QG.models.GNNBlock

@pytest.fixture
def decoder_args():
    return [32, 32]

@pytest.fixture
def decoder_kwargs():
    return {
        "gnn_layer_type": torch_geometric.nn.GCNConv,
        "normalizer_type": torch.nn.LayerNorm,
        "activation_type": torch.nn.SiLU,
        "gnn_layer_args": [],
        "gnn_layer_kwargs": {"add_self_loops": True, "bias": True},
        "norm_args": [32],
        "norm_kwargs": {},
        "skip_args": [32, 32],
        "skip_kwargs": {"weight_initializer": "kaiming_uniform"},
    }

@pytest.fixture
def parent_logit_mlp_type():
    return QG.models.LinearSequential

@pytest.fixture
def parent_logit_mlp_args():
    return [[128, 1]]

@pytest.fixture
def parent_logit_mlp_kwargs():
    return {
        "activation": torch.nn.Identity,
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


def test_minimal_construction(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    """
    Minimal valid decoder configuration:
    - a trivial decoder backbone (LinearSequential)
    - a trivial parent_logit_mlp (producing a single logit)
    This verifies that the class can be instantiated without optional components.
    """
    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    assert isinstance(dec, QG.models.AutoregressiveDecoder)
    assert isinstance(dec.decoder, torch.nn.Module)
    assert isinstance(dec.parent_logit_mlp, torch.nn.Module)
    assert dec.node_feature_decoder is None
    assert dec.ancestor_suppression_strength == 1.0
    assert dec.hidden_dim == 32

@pytest.fixture
def decoder_init_type():
    return QG.models.LinearSequential

@pytest.fixture
def decoder_init_args():
    return [[4, 4]]

@pytest.fixture
def decoder_init_kwargs():
    return {
        "activation": torch.nn.ReLU,
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }

@pytest.fixture
def node_feature_decoder_type():
    return QG.models.LinearSequential

@pytest.fixture
def node_feature_decoder_args():
    return [[4, 4]]

@pytest.fixture
def node_feature_decoder_kwargs():
    return {
        "activation": torch.nn.ReLU,
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }



@pytest.fixture
def minimal_parent_logit_mlp_type():
    return QG.models.LinearSequential

@pytest.fixture
def minimal_parent_logit_mlp_args():
    return [[128, 1]]

@pytest.fixture
def minimal_parent_logit_mlp_kwargs():
    return {
        "activation": torch.nn.Identity,
        "linear_kwargs": [{"bias": True}],
        "activation_kwargs": [{}],
    }


@pytest.fixture
def bad_decoder_args():
    # decoder_args must be a list; here we deliberately provide an invalid type
    return "this_is_not_a_valid_arg_list"




# -------------------------------------------------------------------------
# Realistic parent_logit_mlp
# -------------------------------------------------------------------------

@pytest.fixture
def deep_parent_logit_mlp_type():
    return QG.models.LinearSequential

@pytest.fixture
def deep_parent_logit_mlp_args():
    # MLP with two hidden layers before final scalar output
    return [
        [32 + 32 + 32 + 32, 64],   # concat(h_t, h_prev, z, pos) → 64
        [64, 32],
        [32, 1],                   # scalar output
    ]

@pytest.fixture
def deep_parent_logit_mlp_kwargs():
    return {
        "activation": torch.nn.SiLU,
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
        [32, 64],
        [64, 32],
    ]

@pytest.fixture
def deep_decoder_init_kwargs():
    return {
        "activation": torch.nn.ReLU,
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
        [32, 32],
        [32, 8],   # output 8 node features
    ]

@pytest.fixture
def deep_node_feature_decoder_kwargs():
    return {
        "activation": torch.nn.ReLU,
        "linear_kwargs": [{"bias": True}, {"bias": True}],
        "activation_kwargs": [{}, {}],
    }

@pytest.fixture
def full_decoder_config(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    return {
        "decoder_type": decoder_type,
        "decoder_args": decoder_args,
        "decoder_kwargs": decoder_kwargs,
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
# Full construction with all optional components
# -------------------------------------------------------------------------

def test_full_construction(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    """
    Test that AutoregressiveDecoder can be constructed with optional components:
    - decoder_init
    - node_feature_decoder
    - ancestor suppression strength custom value
    """

    latent_vector = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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

    assert isinstance(dec.decoder, torch.nn.Module)
    assert isinstance(dec.parent_logit_mlp, torch.nn.Module)
    assert isinstance(dec.decoder_init, torch.nn.Module)
    assert isinstance(dec.node_feature_decoder, torch.nn.Module)
    assert dec.ancestor_suppression_strength == 0.7

    # forward in training mode requires teacher forcing
    dec.train()
    L, X, logp = dec(latent_vector, teacher_forcing_targets=teacher_forcing_targets)
    assert L.shape[0] == teacher_forcing_targets.shape[0]
    assert X.shape[0] == teacher_forcing_targets.shape[0]
    assert logp is not None
#
# -------------------------------------------------------------------------
# parent_logit_mlp must output exactly one scalar
# -------------------------------------------------------------------------

def test_parent_logit_mlp_output_must_be_scalar(
    decoder_type, decoder_args, decoder_kwargs
):
    """
    parent_logit_mlp must output exactly one logit per parent.
    If it outputs more than one element, initialization must raise ValueError.
    """
    bad_parent_args = [[16, 2]]  # invalid: outputs 2 logits
    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            decoder_type=decoder_type,
            decoder_args=decoder_args,
            decoder_kwargs=decoder_kwargs,
            parent_logit_mlp_type=QG.models.LinearSequential,
            parent_logit_mlp_args=bad_parent_args,
            parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
        )
 
# -------------------------------------------------------------------------
# latent_dim inference must fail when mlp_in_dim - 3*hidden_dim <= 0
# -------------------------------------------------------------------------

def test_latent_dim_inference_failure(
    decoder_type,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    # hidden_dim = 8 → latent_dim = 16 - 3*8 = -8 → invalid
    bad_decoder_args = [[4, 8]]
    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            decoder_type=decoder_type,
            decoder_args=bad_decoder_args,
            decoder_kwargs=minimal_decoder_kwargs,
            parent_logit_mlp_type=parent_logit_mlp_type,
            parent_logit_mlp_args=parent_logit_mlp_args,
            parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        )

# -------------------------------------------------------------------------
# Config validation: missing required keys
# -------------------------------------------------------------------------

def test_missing_decoder_type_raises(
    minimal_parent_logit_mlp_type,
    minimal_parent_logit_mlp_args,
    minimal_parent_logit_mlp_kwargs,
):
    cfg = {
        # "decoder_type" missing
        "parent_logit_mlp_type": minimal_parent_logit_mlp_type,
        "parent_logit_mlp_args": minimal_parent_logit_mlp_args,
        "parent_logit_mlp_kwargs": minimal_parent_logit_mlp_kwargs,
    }

    with pytest.raises(jsonschema.ValidationError):
        QG.models.AutoregressiveDecoder.from_config(cfg)


def test_missing_parent_logit_mlp_type_raises(
    minimal_decoder_type, minimal_decoder_args, minimal_decoder_kwargs
):
    cfg = {
        "decoder_type": minimal_decoder_type,
        "decoder_args": minimal_decoder_args,
        "decoder_kwargs": minimal_decoder_kwargs,
        # missing "parent_logit_mlp_type"
    }

    with pytest.raises(jsonschema.ValidationError):
        QG.models.AutoregressiveDecoder.from_config(cfg)
        
# -------------------------------------------------------------------------
# malformed decoder args or kwargs must raise schema validation error
# -------------------------------------------------------------------------

def test_malformed_decoder_args_raises(
    minimal_decoder_type,
    bad_decoder_args,
    minimal_decoder_kwargs,
    minimal_parent_logit_mlp_type,
    minimal_parent_logit_mlp_args,
    minimal_parent_logit_mlp_kwargs,
):
    # invalid config: decoder_args must be a list, not a string
    cfg = {
        "decoder_type": minimal_decoder_type,
        "decoder_args": bad_decoder_args,       # malformed
        "decoder_kwargs": minimal_decoder_kwargs,
        "parent_logit_mlp_type": minimal_parent_logit_mlp_type,
        "parent_logit_mlp_args": minimal_parent_logit_mlp_args,
        "parent_logit_mlp_kwargs": minimal_parent_logit_mlp_kwargs,
    }

    with pytest.raises(jsonschema.ValidationError):
        QG.models.AutoregressiveDecoder.from_config(cfg)

# -------------------------------------------------------------------------
# from_config equivalence with standard constructor
# -------------------------------------------------------------------------


def test_from_config_equivalence_full(
    full_decoder_config,
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    # Direct construction
    direct = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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
    assert direct.hidden_dim == 32
    assert cfg_model.hidden_dim == 32

    # Compare parameter structures
    direct_params = {k: v.shape for k, v in direct.named_parameters()}
    cfg_params = {k: v.shape for k, v in cfg_model.named_parameters()}

    assert direct_params == cfg_params
    
 # -------------------------------------------------------------------------
 # training mode requires teacher_forcing_targets
 # -------------------------------------------------------------------------

def test_training_requires_teacher_forcing(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    """
    The decoder must raise ValueError if called in training mode without
    teacher_forcing_targets.
    """
    latent_z = torch.randn(1, 32)
    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()  # training mode enforces teacher forcing

    with pytest.raises(ValueError):
        dec(latent_z, teacher_forcing_targets=None)
 
# -------------------------------------------------------------------------
# forward pass with teacher forcing must succeed
# -------------------------------------------------------------------------

def test_forward_with_teacher_forcing(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_vector_tf = torch.randn(1, 32)

    teacher_forcing_targets_full = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.train()
    L, X, logp = dec(latent_vector_tf, teacher_forcing_targets=teacher_forcing_targets_full)

    assert L.shape[0] == teacher_forcing_targets_full.shape[0]
    assert X is None
    assert logp is not None
 
# -------------------------------------------------------------------------
# eval mode autoregressive sampling must succeed
# -------------------------------------------------------------------------

def test_eval_autoregressive_sampling(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    latent_eval_z = torch.randn(1, 32)
    eval_atom_count = 5
    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    L, X, logp = dec(latent_eval_z, atom_count=eval_atom_count)

    assert L.shape == (eval_atom_count, eval_atom_count)
    assert X is None
    assert logp is None

# -------------------------------------------------------------------------
# parent_logit_mlp input dimension mismatch must fail at forward
# -------------------------------------------------------------------------

def test_parent_mlp_input_dim_mismatch(
    decoder_type, decoder_args, decoder_kwargs
):
    # parent_logit_mlp expects input dim 8 but decoder provides 16
    bad_parent_args = [[8, 1]]

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=bad_parent_args,
        parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
    )

    z = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([[0,0],[1,0]], dtype=torch.float32)

    with pytest.raises(RuntimeError):
        dec.train()
        dec(z, teacher_forcing_targets=teacher_forcing_targets)

# -------------------------------------------------------------------------
# decoder backbone missing required dimensional attribute must raise
# -------------------------------------------------------------------------

def test_decoder_backbone_missing_dim_attr(
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    class BadDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4,4)
        def forward(self, H, edge_index):
            return H

    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            decoder_type=BadDecoder,
            decoder_args=[],
            decoder_kwargs={},
            parent_logit_mlp_type=parent_logit_mlp_type,
            parent_logit_mlp_args=parent_logit_mlp_args,
            parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        )

# -------------------------------------------------------------------------
# ancestor suppression disabled must not modify probabilities
# -------------------------------------------------------------------------

def test_ancestor_suppression_disabled_behavior(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    # Local test inputs
    z = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        ancestor_suppression_strength=0.0,
    )
    dec.train()

    # Capture printed warnings (optional but ensures branch executed)
    L, X, logp = dec(z, teacher_forcing_targets=teacher_forcing_targets)

    # No node features in this configuration
    assert X is None

    # ancestor_suppression flag must be False
    assert dec.ancestor_suppression is False

    # No error and a valid log-likelihood must exist
    assert logp is not None

# -------------------------------------------------------------------------
# ancestor suppression ENABLED must modify probabilities
# -------------------------------------------------------------------------

def test_ancestor_suppression_enabled_behavior(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    # Setup
    z = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
        ancestor_suppression_strength=1.0,
    )
    dec.train()

    # Run forward pass
    L, X, logp = dec(z, teacher_forcing_targets=teacher_forcing_targets)

    # Check presence of log-likelihood
    assert logp is not None

    # Ensure X is None (no node_feature_decoder in this configuration)
    assert X is None

    # ancestor_suppression must be ACTIVE
    assert dec.ancestor_suppression is True

    # Structural correctness of link matrix
    assert L.shape == teacher_forcing_targets.shape

# -------------------------------------------------------------------------
# node_feature_decoder forward pass
# -------------------------------------------------------------------------

def test_node_feature_decoder_forward(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(1, 32)
    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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
    assert X is not None
    assert X.shape[0] == teacher_forcing_targets.shape[0]
    # Must have positive feature dimension
    assert X.shape[1] > 0
    # log-likelihood must still be computed
    assert logp is not None


# -------------------------------------------------------------------------
# reconstruct_node_features() must return (L, X) when configured
# -------------------------------------------------------------------------

def test_reconstruct_node_features(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    z = torch.randn(1, 32)
    atom_count = 4

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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

    # X must exist because node_feature_decoder is configured
    assert X is not None
    assert X.shape[0] == atom_count
    assert X.shape[1] > 0  # must have at least one feature dimension


# -------------------------------------------------------------------------
# reconstruct_node_features must raise without node_feature_decoder
# -------------------------------------------------------------------------

def test_reconstruct_node_features_without_decoder_raises(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(1, 32)
    atom_count = 4

    # Construct decoder WITHOUT node_feature_decoder
    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    # Must raise because node_feature_decoder is None
    with pytest.raises(RuntimeError):
        dec.reconstruct_node_features(z, atom_count)


# -------------------------------------------------------------------------
# eval mode without teacher_forcing requires atom_count
# -------------------------------------------------------------------------

def test_eval_requires_atom_count_when_no_teacher_forcing(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(1, 32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
        parent_logit_mlp_type=parent_logit_mlp_type,
        parent_logit_mlp_args=parent_logit_mlp_args,
        parent_logit_mlp_kwargs=parent_logit_mlp_kwargs,
    )

    dec.eval()

    # No teacher forcing and no atom_count → must raise
    import pytest
    with pytest.raises(ValueError):
        dec(z, teacher_forcing_targets=None, atom_count=None)
#
# -------------------------------------------------------------------------
# atom_count must be ignored when teacher_forcing_targets is provided
# -------------------------------------------------------------------------

def test_atom_count_ignored_when_teacher_forcing_given(
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs
):
    z = torch.randn(1, 32)

    teacher_forcing_targets = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=torch.float32)

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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

    # logp must be computed in training mode
    assert logp is not None

    # node features are disabled here
    assert X is None
#
# -------------------------------------------------------------------------
# save/load equivalence test
# -------------------------------------------------------------------------

def test_save_and_load_equivalence(
    tmp_path,
    full_decoder_config,
    decoder_type, decoder_args, decoder_kwargs,
    parent_logit_mlp_type, parent_logit_mlp_args, parent_logit_mlp_kwargs,
    decoder_init_type, decoder_init_args, decoder_init_kwargs,
    node_feature_decoder_type, node_feature_decoder_args, node_feature_decoder_kwargs
):
    # Build model
    dec = QG.models.AutoregressiveDecoder(
        decoder_type=decoder_type,
        decoder_args=decoder_args,
        decoder_kwargs=decoder_kwargs,
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

        # -------------------------------------------------------------------------
# GNN-based integration tests (minimal but sufficient)
# -------------------------------------------------------------------------

def test_gnn_decoder_instantiation(
    gnn_decoder_type, gnn_decoder_args, gnn_decoder_kwargs,
    deep_parent_logit_mlp_type, deep_parent_logit_mlp_args, deep_parent_logit_mlp_kwargs,
    deep_decoder_init_type, deep_decoder_init_args, deep_decoder_init_kwargs,
    deep_node_feature_decoder_type, deep_node_feature_decoder_args, deep_node_feature_decoder_kwargs,
):
    """
    Integration test verifying that a realistic GNN-based decoder
    (GNNBlock backbone + multi-layer MLPs) instantiates correctly.
    No forward pass is tested here.
    """

    dec = QG.models.AutoregressiveDecoder(
        decoder_type=gnn_decoder_type,
        decoder_args=gnn_decoder_args,
        decoder_kwargs=gnn_decoder_kwargs,

        parent_logit_mlp_type=deep_parent_logit_mlp_type,
        parent_logit_mlp_args=deep_parent_logit_mlp_args,
        parent_logit_mlp_kwargs=deep_parent_logit_mlp_kwargs,

        decoder_init_type=deep_decoder_init_type,
        decoder_init_args=deep_decoder_init_args,
        decoder_init_kwargs=deep_decoder_init_kwargs,

        node_feature_decoder_type=deep_node_feature_decoder_type,
        node_feature_decoder_args=deep_node_feature_decoder_args,
        node_feature_decoder_kwargs=deep_node_feature_decoder_kwargs,

        ancestor_suppression_strength=0.5,
    )

    # Basic structural assertions
    assert isinstance(dec, QG.models.AutoregressiveDecoder)
    assert dec.hidden_dim == 32  # inferred from GNNBlock out_dim
    assert isinstance(dec.decoder, torch.nn.Module)
    assert isinstance(dec.parent_logit_mlp, torch.nn.Module)
    assert isinstance(dec.decoder_init, torch.nn.Module)
    assert isinstance(dec.node_feature_decoder, torch.nn.Module)
    assert dec.ancestor_suppression_strength == 0.5