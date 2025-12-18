import pytest


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
        "redraw_interval": 1000,
    }


def test_redraw_projection_instantiate():
    assert 3 == 6


def test_redraw_projection_forward():
    assert 3 == 6


def test_gps_transformer_instantiate():
    assert 3 == 6


def test_gps_transformer_forward():
    assert 3 == 6


def test_gps_model_instantiate():
    assert 3 == 6


def test_gps_model_fromconfig():
    assert 3 == 6


def test_gps_model_forward():
    assert 3 == 6
