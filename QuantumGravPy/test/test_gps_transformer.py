import pytest


pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from QuantumGrav.models.gps_transformer import (
    RedrawProjection,
    GPSModel,
    GPSTransformer,
)


@pytest.fixture
def config():
    return {
        "in_features": 2,
        "out_features": 4,
        "channels": 8,
        "num_heads": 2,
        "num_layers": 2,
        "attn_type": "performer",
        "attn_kwargs": {},
        "redraw_interval": 3,  # small to exercise redraw logic quickly
    }


def test_redraw_projection_instantiate():
    dummy = torch.nn.Module()
    rp = RedrawProjection(dummy, redraw_interval=5)
    assert rp.model is dummy
    assert rp.redraw_interval == 5
    assert rp.num_last_redraw == 0


def test_redraw_projection_forward():
    # Create a simple dummy model to exercise counter logic without performer modules
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.training = False

    m = Dummy()
    rp = RedrawProjection(m, redraw_interval=2)

    # Not training: no change
    rp.redraw_projections()
    assert rp.num_last_redraw == 0

    # Training but interval None: no change
    m.training = True
    rp_none = RedrawProjection(m, redraw_interval=None)
    rp_none.redraw_projections()
    assert rp_none.num_last_redraw == 0

    # Training with interval: counter increments, then resets when threshold reached
    rp.redraw_projections()  # -> 1
    assert rp.num_last_redraw == 1
    rp.redraw_projections()  # -> 2
    assert rp.num_last_redraw == 2
    rp.redraw_projections()  # >= interval -> reset to 0
    assert rp.num_last_redraw == 0


def test_gps_transformer_instantiate(config):
    model = GPSTransformer(**config)
    # Basic structure
    assert isinstance(model.transformer_model, GPSModel)
    assert len(model.transformer_model.convs) == config["num_layers"]
    # Redraw helper configured for performer
    assert model.transformer_model.redraw.redraw_interval == config["redraw_interval"]

    # Non-performer disables redraw regardless of provided interval
    cfg2 = {**config, "attn_type": "multihead", "redraw_interval": 2}
    model2 = GPSTransformer(**cfg2)
    assert model2.transformer_model.redraw.redraw_interval is None


def _toy_graph(in_features: int):
    # Two small graphs of three nodes each, total N=6
    x = torch.randn(6, in_features)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],  # src
            [1, 0, 2, 1, 4, 3, 5, 4],  # dst
        ],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return x, edge_index, batch


def test_gps_transformer_forward(config, monkeypatch):
    model = GPSTransformer(**config)
    model.train()

    # Spy on redraw_projections to ensure it's called by forward()
    called = {"n": 0}

    def _spy():
        called["n"] += 1

    monkeypatch.setattr(model.transformer_model.redraw, "redraw_projections", _spy)

    x, edge_index, batch = _toy_graph(config["in_features"])
    out = model(x, edge_index, batch)

    assert out.shape == (batch.max().item() + 1, config["out_features"])
    assert called["n"] == 1  # one call per forward


def test_gps_model_instantiate(config):
    m = GPSModel(
        config["in_features"],
        config["out_features"],
        config["channels"],
        config["num_heads"],
        config["num_layers"],
        config["attn_type"],
        config["attn_kwargs"],
        redraw_interval=config["redraw_interval"],
    )
    assert len(m.convs) == config["num_layers"]
    assert isinstance(m.input_proj, torch.nn.Linear)
    assert isinstance(m.mlp, torch.nn.Sequential)


def test_gps_model_fromconfig(config):
    # Note: from_config is defined on GPSTransformer
    model = GPSTransformer.from_config(config)
    assert isinstance(model, GPSTransformer)
    assert len(model.transformer_model.convs) == config["num_layers"]


def test_gps_model_forward(config):
    # Also exercise redraw interval branch inside real forward path
    # Use small interval so we hit the redraw branch after 3 forwards
    cfg = {**config, "redraw_interval": 2}
    model = GPSTransformer.from_config(cfg)
    model.train()

    # Patch PerformerAttention.redraw_projection_matrix to observe calls
    import torch_geometric

    called = {"n": 0}
    PA = torch_geometric.nn.attention.PerformerAttention
    orig = PA.redraw_projection_matrix

    def counted(self):  # type: ignore[override]
        called["n"] += 1
        # call original if exists to avoid breaking internals
        return orig(self)

    try:
        # Only patch if attn_type is performer

        # apply monkeypatching via setattr to the class
        setattr(PA, "redraw_projection_matrix", counted)

        x, edge_index, batch = _toy_graph(cfg["in_features"])
        # Three forwards to cross the >= interval threshold
        model(x, edge_index, batch)
        model(x, edge_index, batch)
        model(x, edge_index, batch)

        # We should have reset the counter on the third call
        assert model.transformer_model.redraw.num_last_redraw == 0
        # And at least one performer attention module should have been redrawn
        assert called["n"] >= 1

        # Now in eval mode, the counter should not change
        model.eval()
        before = model.transformer_model.redraw.num_last_redraw
        model(x, edge_index, batch)
        assert model.transformer_model.redraw.num_last_redraw == before
    finally:
        # restore original to avoid side effects
        setattr(PA, "redraw_projection_matrix", orig)
