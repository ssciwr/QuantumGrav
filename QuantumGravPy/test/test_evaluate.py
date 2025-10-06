import logging
import QuantumGrav as QG
from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
import pytest


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0], data.y.to(torch.float32))
    return loss


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
        assert evaluator.data == [
            (expected_avg, expected_std),
        ]

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


def test_default_early_stopping_creation(input):
    """Test the DefaultEarlyStopping class."""

    early_stopping = QG.DefaultEarlyStopping(
        input["patience"],
        input["delta"],
        input["window"],
        metric=input["metric"],
        smoothing=input["smoothing"],
        grace_period=input["grace_period"],
        init_best_score=42,
        mode="all",
    )

    assert early_stopping.patience == input["patience"]
    assert early_stopping.delta == input["delta"]
    assert early_stopping.window == input["window"]
    assert early_stopping.best_score == [42, 42]
    assert early_stopping.current_patience == input["patience"]
    assert early_stopping.metric == ["loss", "other_loss"]
    assert early_stopping.smoothing is False
    assert early_stopping.found_better == [False, False]
    assert early_stopping.logger is not None
    assert early_stopping.mode == "all"

    early_stopping_callable = QG.DefaultEarlyStopping(
        input["patience"],
        input["delta"],
        input["window"],
        metric=input["metric"],
        smoothing=input["smoothing"],
        grace_period=input["grace_period"],
        init_best_score=42,
        mode=lambda b: all(b),
    )

    assert callable(early_stopping_callable.mode)
    with pytest.raises(
        ValueError, match="Inconsistent lengths for early stopping parameters"
    ):
        early_stopping_callable = QG.DefaultEarlyStopping(
            input["patience"],
            input["delta"],
            [12, 7, 9],
            metric=input["metric"],
            smoothing=input["smoothing"],
            grace_period=input["grace_period"],
            init_best_score=42,
            mode=lambda b: all(b),
        )


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_check(smoothed, input):
    """Test the check method of DefaultEarlyStopping."""
    early_stopping = QG.DefaultEarlyStopping(
        input["patience"], input["delta"], input["window"], smoothing=smoothed
    )
    early_stopping.best_score = [0.06, 0.06]
    losses = pd.DataFrame(
        {
            "loss": 2
            * [
                0.1,
                0.09,
                0.08,
                0.07,
            ]
        }
    )

    assert early_stopping(losses) is False
    assert early_stopping.current_patience == 1


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_found_better_works(smoothed, input):
    early_stopping = QG.DefaultEarlyStopping(
        patience,
        delta,
        window,
        metric=metric,
        smoothing=smoothing,
        grace_period=grace_period,
    )
    early_stopping.best_score = [0.09, 0.2]
    early_stopping.current_patience = 1
    losses = pd.DataFrame(
        {
            "loss": [
                0.1,
                0.09,
                0.08,
                0.07,
            ],
            "other_loss": [0.2, 0.3, 0.4, 0.5],
        }
    )

    assert early_stopping(losses) is False
    assert early_stopping.current_patience == 7

    losses = pd.DataFrame(
        {
            "loss": [
                0.0001,
                0.0001,
            ],
            "other_loss": [
                0.0001,
                0.0001,
            ],
        }
    )
    assert early_stopping(losses) is False
    assert early_stopping.current_patience == 7
    assert early_stopping.found_better == [True, True]
    assert early_stopping.best_score == [0.0001, 0.0001]


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_triggered(smoothed):
    patience = 1
    delta = 1e-4
    window = 5

    early_stopping = QG.DefaultEarlyStopping(
        patience, delta, window, smoothing=smoothed
    )
    early_stopping.best_score = 0.09
    early_stopping.current_patience = 1
    losses = pd.DataFrame(
        {
            "loss": 2
            * [
                1.1,
                1.09,
                1.08,
                1.07,
            ]
        }
    )

    assert early_stopping(losses) is True
    assert early_stopping.best_score == 0.09  # Best score should be updated
    assert early_stopping.current_patience == 0


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_criterion(smoothed):
    patience = 1
    delta = 1e-4
    window = 5
    early_stopping = QG.DefaultEarlyStopping(
        patience,
        delta,
        window,
        smoothing=smoothed,
        criterion=lambda x, d: bool(d[x.metric].mean() < 2),
    )
    early_stopping.best_score = 0.09
    early_stopping.current_patience = 1
    losses = pd.DataFrame(
        {
            "loss": 2
            * [
                1.1,
                1.09,
                1.08,
                1.07,
            ]
        }
    )

    assert early_stopping(losses) is True
    assert early_stopping.best_score == 0.09  # Best score should be updated
    assert early_stopping.current_patience == 0
