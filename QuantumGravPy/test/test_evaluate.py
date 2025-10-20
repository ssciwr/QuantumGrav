import logging
import QuantumGrav as QG
from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
import pytest
from jsonschema import ValidationError


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0].unsqueeze(0), data.y.to(torch.float32))
    return loss


@pytest.fixture(scope="session")
def input():
    return {
        "patience": 7,
        "delta": [1e-2, 1e-4],
        "window": [12, 7],
        "metric": ["loss", "other_loss"],
        "smoothing": False,
        "grace_period": [8, 4],
    }


@pytest.fixture(scope="session")
def f1eval_config():
    return {"average": "macro", "labels": [0, 1]}


@pytest.fixture(scope="session")
def f1eval_config_broken():
    return {"average": "not_recognized", "labels": [0, 1]}


def test_f1eval_creation(f1eval_config):
    evaluator = QG.evaluate.F1ScoreEval(
        average=f1eval_config["average"],
        labels=f1eval_config["labels"],
    )
    assert evaluator.average == f1eval_config["average"]
    assert evaluator.labels == f1eval_config["labels"]


def test_f1eval_fromconfig(f1eval_config):
    evaluator = QG.evaluate.F1ScoreEval.from_config(f1eval_config)
    assert evaluator.average == f1eval_config["average"]
    assert evaluator.labels == f1eval_config["labels"]


def test_f1eval_fromconfig_broken(f1eval_config_broken):
    with pytest.raises(ValidationError):
        QG.evaluate.F1ScoreEval.from_config(f1eval_config_broken)


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
def test_default_early_stopping_check_positive(smoothed, input):
    """Test the check method of DefaultEarlyStopping - positive."""
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=smoothed,
    )

    early_stopping.best_score = [12.0, 12.0]
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
    assert early_stopping.current_patience == input["patience"]
    assert early_stopping.current_grace_period == [
        input["grace_period"][0] - 1,
        input["grace_period"][1] - 1,
    ]
    assert early_stopping.found_better == [True, True]


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_check_positive_with_untracked_metric(smoothed, input):
    """Test the check method of DefaultEarlyStopping - positive."""
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=smoothed,
    )

    early_stopping.best_score = [12.0, 12.0]
    losses = pd.DataFrame(
        {
            "loss": [
                0.1,
                0.09,
                0.08,
                0.07,
            ],
            "untracked": [0.2, 0.3, 0.4, 0.5],
        }
    )

    assert early_stopping(losses) is False
    assert early_stopping.current_patience == input["patience"]
    assert early_stopping.current_grace_period == [
        input["grace_period"][0] - 1,
        input["grace_period"][1] - 1,
    ]
    assert early_stopping.found_better == [True, True]


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_check_negative(smoothed, input):
    """Test the check method of DefaultEarlyStopping - negative."""
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=smoothed,
    )

    early_stopping.best_score = [1e-5, 1e-5]
    # set grace period artificially lower
    early_stopping.current_grace_period = [0, 0]
    losses = pd.DataFrame(
        {
            "loss": [
                10.0,
                10.0,
            ],
            "other_loss": [
                10.0,
                10.0,
            ],
        }
    )
    assert early_stopping(losses) is False
    assert early_stopping.current_patience == input["patience"] - 1
    assert np.array_equal(early_stopping.current_grace_period, [0, 0])
    assert np.array_equal(early_stopping.best_score, [1e-5, 1e-5])  # not updated


@pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
def test_default_early_stopping_triggered(smoothed, input):
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=smoothed,
    )
    early_stopping.best_score = [1e-4, 1e-4]
    early_stopping.patience = 1
    early_stopping.grace_period = [0, 0]
    early_stopping.reset()

    losses = pd.DataFrame(
        {
            "loss": [
                0.1,
                0.1,
            ],
            "other_loss": [
                0.1,
                0.1,
            ],
        }
    )
    assert early_stopping(losses) is True
    assert early_stopping.current_patience == 0


def test_default_early_stopping_add_task(input):
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=False,
    )

    early_stopping.add_task(1e-3, 12, "blergh", 3)

    assert early_stopping.delta[-1] == 1e-3
    assert early_stopping.window[-1] == 12
    assert early_stopping.metric[-1] == "blergh"
    assert early_stopping.grace_period[-1] == 3
    assert early_stopping.current_grace_period[-1] == 3
    assert early_stopping.best_score[-1] == np.inf
    assert early_stopping.found_better == [False, False, False]


def test_default_early_stopping_reset(input):
    early_stopping = QG.DefaultEarlyStopping(
        patience=input["patience"],
        delta=input["delta"],
        window=input["window"],
        metric=input["metric"],
        grace_period=input["grace_period"],
        smoothing=False,
    )

    early_stopping.current_patience = input["patience"] - 2
    early_stopping.current_grace_period = [
        input["grace_period"][0] - 1,
        input["grace_period"][1] - 1,
    ]
    early_stopping.found_better = [True, False, False]
    early_stopping.best_score = [1, 2, 3]
    early_stopping.reset()

    assert early_stopping.current_patience == input["patience"]
    assert early_stopping.current_grace_period == input["grace_period"]
    assert early_stopping.found_better == [False] * len(input["metric"])
    assert early_stopping.best_score == [1, 2, 3]
    assert early_stopping.current_grace_period == early_stopping.grace_period
    assert early_stopping.current_patience == early_stopping.patience
