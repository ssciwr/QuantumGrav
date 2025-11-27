import logging
import QuantumGrav as QG
from torch_geometric.data import Data
import torch
import torch_geometric
import numpy as np
import pytest


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0].unsqueeze(0), data.y.to(torch.float32))
    return loss


# Helper functions for aggregation
def concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, y), dim=1)


def cat1(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(xs, dim=1)


@pytest.fixture
def model_config_eval():
    config = {
        "encoder_type": QG.models.GNNBlock,
        "encoder_args": [2, 32],
        "encoder_kwargs": {
            "dropout": 0.3,
            "gnn_layer_type": torch_geometric.nn.conv.GCNConv,
            "normalizer_type": torch.nn.BatchNorm1d,
            "activation_type": torch.nn.ReLU,
            "gnn_layer_args": [],
            "gnn_layer_kwargs": {"cached": False, "bias": True, "add_self_loops": True},
            "norm_args": [32],
            "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
            "skip_args": [2, 32],
            "skip_kwargs": {"weight_initializer": "kaiming_uniform"},
        },
        # After two pooling ops concatenated along dim=1, encoder output 32 -> 64 input to heads
        "downstream_tasks": [
            [
                QG.models.LinearSequential,
                [
                    [(64, 24), (24, 18), (18, 2)],
                    [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
                ],
                {
                    "linear_kwargs": [
                        {"bias": True},
                        {"bias": True},
                        {"bias": False},
                    ],
                    "activation_kwargs": [{"inplace": False}, {}, {}],
                },
            ],
            [
                QG.models.LinearSequential,
                [
                    [(64, 24), (24, 18), (18, 3)],
                    [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
                ],
                {
                    "linear_kwargs": [
                        {"bias": True},
                        {"bias": True},
                        {"bias": False},
                    ],
                    "activation_kwargs": [{"inplace": False}, {}, {}],
                },
            ],
        ],
        "pooling_layers": [
            [torch_geometric.nn.global_mean_pool, [], {}],
            [torch_geometric.nn.global_max_pool, [], {}],
        ],
        "aggregate_pooling_type": cat1,
        "active_tasks": {0: True, 1: True},
    }
    return config


@pytest.fixture
def gnn_model_eval(model_config_eval):
    """Fixture to create a GNNModel for evaluation."""

    model = QG.GNNModel.from_config(
        model_config_eval,
    )
    model.eval()
    return model


@pytest.fixture
def input():
    return {
        "patience": 7,
        "delta": [1e-2, 1e-4],
        "window": [12, 7],
        "metric": ["loss", "other_loss"],
        "smoothing": False,
        "grace_period": [8, 4],
    }


def test_default_evaluator_creation(gnn_model_eval):
    """Test the DefaultEvaluator class."""
    device = torch.device("cpu")
    evaluator = QG.DefaultEvaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda data: gnn_model_eval(data, data.edge_index, data.batch),
    )

    assert evaluator.device == device
    assert evaluator.criterion is compute_loss
    assert evaluator.apply_model is not None


def test_default_evaluator_evaluate(make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    device = torch.device("cpu")
    model = gnn_model_eval.to(device)
    evaluator = QG.DefaultEvaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    losses = evaluator.evaluate(model, dataloader)
    assert len(losses) == len(dataloader)
    assert torch.Tensor(losses).dtype == torch.float32


def test_default_evaluator_report(caplog):
    device = torch.device("cpu")
    evaluator = QG.DefaultEvaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
    )
    assert len(evaluator.data) == 0
    losses = np.random.rand(100)
    expected_avg = np.mean(losses)
    expected_std = np.std(losses)

    with caplog.at_level(logging.INFO):
        evaluator.report(losses)
        # Test specific content
        assert f"Average loss: {expected_avg}" in caplog.text
        assert f"Standard deviation: {expected_std}" in caplog.text


def test_default_tester_creation():
    """Test the DefaultTester class."""
    device = torch.device("cpu")
    tester = QG.DefaultTester(
        device=device,
        criterion=compute_loss,
        apply_model=None,
    )

    assert tester.device == device
    assert tester.criterion is compute_loss
    assert tester.apply_model is None


def test_default_tester_test(make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    device = torch.device("cpu")
    tester = QG.DefaultTester(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    testdata = tester.test(gnn_model_eval, dataloader)
    assert len(testdata) == len(dataloader)
    assert torch.Tensor(testdata).dtype == torch.float32


def test_default_tester_report(caplog):
    device = torch.device("cpu")
    tester = QG.DefaultTester(
        device=device,
        criterion=compute_loss,
        apply_model=None,
    )
    losses = np.random.rand(100)
    expected_avg = np.mean(losses)
    expected_std = np.std(losses)

    with caplog.at_level(logging.INFO):
        tester.report(losses)
        # Test specific content
        assert f"Average loss: {expected_avg}" in caplog.text
        assert f"Standard deviation: {expected_std}" in caplog.text
