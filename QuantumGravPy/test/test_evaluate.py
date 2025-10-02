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


def test_default_early_stopping_creation():
    """Test the DefaultEarlyStopping class."""
    patience = 5
    delta = 1e-4
    window = 7

    early_stopping = QG.DefaultEarlyStopping(patience, delta, window)

    assert early_stopping.patience == patience
    assert early_stopping.delta == delta
    assert early_stopping.window == window
    assert early_stopping.best_score == np.inf
    assert early_stopping.current_patience == patience
    assert early_stopping.metric == "loss"
    assert early_stopping.smoothing is False
    assert early_stopping.found_better is False
    assert early_stopping.logger is not None


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_check(smoothed):
    """Test the check method of DefaultEarlyStopping."""
    patience = 2
    delta = 1e-4
    window = 5

    early_stopping = QG.DefaultEarlyStopping(
        patience, delta, window, smoothing=smoothed
    )
    early_stopping.best_score = float("inf")

    early_stopping.best_score = 0.06
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
    assert early_stopping.best_score == 0.06


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_reset(smoothed):
    patience = 2
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
                0.1,
                0.09,
                0.08,
                0.07,
            ]
        }
    )

    assert early_stopping(losses) is False
    assert early_stopping.best_score < 0.09  # Best score should be updated
    assert early_stopping.current_patience == 2


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
