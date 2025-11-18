import QuantumGrav as QG
import torch_geometric
import torch


def test_skipconnection_construction():
    skip = QG.SkipConnection(
        4, 32, weight_initializer="glorot", bias_initializer="zeros"
    )
    assert isinstance(skip.proj, torch_geometric.nn.dense.Linear)
    assert skip.proj.in_channels == 4
    assert skip.proj.out_channels == 32


def test_skipconnection_run():
    x = torch.rand(32)
    skip = QG.SkipConnection(
        4, 32, weight_initializer="glorot", bias_initializer="zeros"
    )
    x_old = torch.rand(4)
    y = skip.forward(x_old, x)

    assert y.shape == (32,)
    assert x.shape == (32,)
    assert x_old.shape == (4,)
