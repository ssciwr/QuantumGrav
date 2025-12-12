# -------------------------------------------------------------------------
# GRU args must specify at least two elements: [input_dim, hidden_dim]
# -------------------------------------------------------------------------

def test_gru_args_too_short(gru_type, gru_kwargs):
    """
    NodeUpdateGRU must receive gru_args with at least two integers:
        gru_args[0] → input_dim
        gru_args[1] → hidden_dim
    Providing fewer must raise ValueError.
    """
    bad_gru_args = [32]  # missing hidden_dim

    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=QG.models.LinearSequential,
            parent_logit_mlp_args=[
                [[128, 1]],
                [torch.nn.Identity],
            ],
            parent_logit_mlp_kwargs={
                "linear_kwargs": [{"bias": True}],
                "activation_kwargs": [{}],
            },
        )

# -------------------------------------------------------------------------
# GRU args must contain valid integer in_dim and hidden_dim
# -------------------------------------------------------------------------

def test_gru_args_non_integer(
    gru_type, gru_kwargs,
    minimal_parent_logit_mlp_type, minimal_parent_logit_mlp_args, minimal_parent_logit_mlp_kwargs
):
    """
    NodeUpdateGRU requires that the first two elements of gru_args are
    integer input_dim and hidden_dim. Non‑integer values must raise ValueError.
    """
    bad_gru_args = ["not_an_int", 32]  # invalid input_dim

    with pytest.raises(TypeError):
        QG.models.AutoregressiveDecoder(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=minimal_parent_logit_mlp_type,
            parent_logit_mlp_args=minimal_parent_logit_mlp_args,
            parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
        )

    bad_gru_args = [32, "not_an_int"]  # invalid hidden_dim

    with pytest.raises(TypeError):
        QG.models.AutoregressiveDecoder(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=minimal_parent_logit_mlp_type,
            parent_logit_mlp_args=minimal_parent_logit_mlp_args,
            parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
        )

# -------------------------------------------------------------------------
# GRU args must contain positive integer in_dim and hidden_dim
# -------------------------------------------------------------------------

def test_gru_args_non_positive(
    gru_type, gru_kwargs,
    minimal_parent_logit_mlp_type, minimal_parent_logit_mlp_args, minimal_parent_logit_mlp_kwargs
):
    """
    NodeUpdateGRU requires the first two gru_args to be strictly positive integers.
    Zero or negative values must raise ValueError.
    """

    # Case 1: input_dim <= 0
    bad_gru_args = [0, 32]
    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=minimal_parent_logit_mlp_type,
            parent_logit_mlp_args=minimal_parent_logit_mlp_args,
            parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
        )

    # Case 2: hidden_dim <= 0
    bad_gru_args = [32, -1]
    with pytest.raises(ValueError):
        QG.models.AutoregressiveDecoder(
            gru_type=gru_type,
            gru_args=bad_gru_args,
            gru_kwargs=gru_kwargs,
            parent_logit_mlp_type=minimal_parent_logit_mlp_type,
            parent_logit_mlp_args=minimal_parent_logit_mlp_args,
            parent_logit_mlp_kwargs=minimal_parent_logit_mlp_kwargs,
        )