import logging
import QuantumGrav as QG
from torch_geometric.data import Data
import torch
import numpy as np


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


def test_default_early_stopping_check(caplog):
    """Test the check method of DefaultEarlyStopping."""
    patience = 2
    delta = 1e-4
    window = 3

    early_stopping = QG.DefaultEarlyStopping(patience, delta, window)
    early_stopping.best_score = float("inf")

    # Simulate some validation losses
    losses = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]

    with caplog.at_level(logging.INFO):
        assert early_stopping(losses) is False  # Not enough data to decide
        assert early_stopping.best_score != float("inf")  # Best score should be updated
        assert early_stopping.current_patience == 2
        assert (
            "Early stopping patience reset: 2 -> 2, early stopping best score updated: "
            in caplog.text
        )

    working_losses = [2 * x for x in losses]

    with caplog.at_level(logging.INFO):
        assert early_stopping(working_losses) is False  # Should trigger early stoppin
        assert early_stopping.current_patience == 1  # Patience should be reset

        assert "Early stopping patience decreased:" in caplog.text

        assert early_stopping(working_losses) is True  # Should trigger early stoppin

        assert early_stopping.current_patience == 0  # Patience should be exhausted
        assert "Early stopping patience decreased: 1 -> 0" in caplog.text


def test_f1_evaluator_creation(gnn_model_eval):
    """Test the F1Evaluator class."""
    device = torch.device("cpu")
    evaluator = QG.F1Evaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
        prefix="test",
    )

    assert evaluator.device == device
    assert evaluator.apply_model is not None


def test_f1_evaluator_evaluate(make_dataloader, gnn_model_eval):
    "test F1Evaluator.evaluate() method"
    dataloader = make_dataloader
    device = torch.device("cpu")
    model = gnn_model_eval.to(device)
    evaluator = QG.F1Evaluator(
        device=device,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    f1_scores = evaluator.evaluate(model, dataloader)
    assert len(f1_scores) == len(dataloader)
    assert torch.Tensor(f1_scores).dtype == torch.float32


def test_f1_evaluator_report(caplog):
    "test F1Evaluator.report() method"
    device = torch.device("cpu")
    evaluator = QG.F1Evaluator(
        device=device,
        apply_model=None,
    )
    assert len(evaluator.data) == 0
    f1_scores = np.random.rand(100)
    expected_avg = np.mean(f1_scores)
    expected_std = np.std(f1_scores)

    with caplog.at_level(logging.INFO):
        evaluator.report(f1_scores)
        assert evaluator.data == [
            (expected_avg, expected_std),
        ]

        # Test specific content
        assert f"Average F1 score: {expected_avg}" in caplog.text
        assert f"Standard deviation: {expected_std}" in caplog.text
        # TODO: add more tests for F1Evaluator, F1Validator, F1Tester, AccuracyEvaluator, AccuracyValidator, AccuracyTester


def test_accuracy_evaluator_creation(gnn_model_eval):
    "test AccuracyEvaluator creation"
    """Test the AccuracyEvaluator class."""
    device = torch.device("cpu")
    evaluator = QG.AccuracyEvaluator(
        device=device,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )

    assert evaluator.device == device
    assert evaluator.apply_model is not None


def test_accuracy_evaluator_evaluate(make_dataloader, gnn_model_eval):
    "test AccuracyEvaluator.evaluate() method"
    dataloader = make_dataloader
    device = torch.device("cpu")
    model = gnn_model_eval.to(device)
    evaluator = QG.AccuracyEvaluator(
        device=device,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    accuracies = evaluator.evaluate(model, dataloader)
    assert len(accuracies) == len(dataloader)
    assert torch.Tensor(accuracies).dtype == torch.float32


def test_accuracy_evaluator_report(caplog):
    "test AccuracyEvaluator.report() method"
    device = torch.device("cpu")
    evaluator = QG.AccuracyEvaluator(
        device=device,
        apply_model=None,
    )
    assert len(evaluator.data) == 0
    accuracies = np.random.rand(100)
    expected_avg = np.mean(accuracies)
    expected_std = np.std(accuracies)

    with caplog.at_level(logging.INFO):
        evaluator.report(accuracies)
        assert evaluator.data == [
            (expected_avg, expected_std),
        ]

        # Test specific content
        assert f"Average accuracy: {expected_avg}" in caplog.text
        assert f"Standard deviation: {expected_std}" in caplog.text
        # TODO: add more output tests
