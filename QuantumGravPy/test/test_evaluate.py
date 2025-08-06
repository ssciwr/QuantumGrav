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
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
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
        apply_model=None,
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
