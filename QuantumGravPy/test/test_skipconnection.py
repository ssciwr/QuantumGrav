import torch
import torch_geometric

import QuantumGrav as QG


def test_creation_works():
    skip = QG.models.SkipConnection(
        12,
        12,
        weight_initializer="glorot",
    )
    assert isinstance(skip.proj, torch.nn.Identity)

    skip_p = QG.models.SkipConnection(
        12,
        24,
    )
    assert isinstance(skip_p.proj, torch_geometric.nn.dense.Linear)


def test_forward():
    skip = QG.models.SkipConnection(
        12,
        12,
        weight_initializer="glorot",
    )
    x = torch.rand(12)
    x_s = skip.forward(x, x)
    assert x_s.shape == (12,)

    skip_p = QG.models.SkipConnection(
        12,
        24,
    )
    x_res = torch.rand(24)
    x_s = skip_p.forward(x, x_res)
    assert x_s.shape == (24,)


def test_to_config():
    skip = QG.models.SkipConnection(
        12,
        24,
        weight_initializer="uniform",
    )
    cfg = skip.to_config()
    assert cfg["in_channels"] == 12
    assert cfg["out_channels"] == 24
    assert cfg["weight_initializer"] == "uniform"


def test_from_config():
    cfg = {
        "in_channels": 12,
        "out_channels": 24,
        "weight_initializer": "uniform",
    }

    skip = QG.models.SkipConnection.from_config(cfg)
    assert hasattr(skip, "proj") is True
