import QuantumGrav as QG
from torch_geometric.data import Data
import torch


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0], data.y.to(torch.float32))
    return loss


def test_default_evaluator_creation(make_dataloader, gnn_model_eval):
    """Test the DefaultEvaluator class."""
    device = torch.device("cpu")
    evaluator = QG.DefaultEvaluator(
        device=device,
        criterion=compute_loss,
        apply_model=None,
    )

    assert evaluator.device == device
    assert evaluator.criterion is compute_loss
    assert evaluator.apply_model is None


def test_default_evaluator_evaluate(evaluator, make_dataloader, gnn_model_eval):
    # dataloader = make_dataloader
    # losses = evaluator.evaluate(gnn_model_eval, dataloader)

    assert 3 == 6


def test_default_evaluator_report(evaluator, make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    losses = evaluator.evaluate(gnn_model_eval, dataloader)
    evaluator.report(losses)

    assert 3 == 6


def test_default_tester_creation(make_dataloader, gnn_model_eval):
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


def test_default_tester_test(tester, make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    tester.test(gnn_model_eval, dataloader)

    assert 3 == 6


def test_default_tester_report(tester, make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    losses = tester.test(gnn_model_eval, dataloader)
    tester.report(losses)

    assert 3 == 6


def test_default_validator_creation(make_dataloader, gnn_model_eval):
    """Test the DefaultValidator class."""
    device = torch.device("cpu")
    validator = QG.DefaultValidator(
        device=device,
        criterion=compute_loss,
        apply_model=None,
    )

    assert validator.device == device
    assert validator.criterion is compute_loss
    assert validator.apply_model is None


def test_default_validator_validate(validator, make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    validator.validate(gnn_model_eval, dataloader)

    assert 3 == 6


def test_default_validator_report(validator, make_dataloader, gnn_model_eval):
    dataloader = make_dataloader
    losses = validator.validate(gnn_model_eval, dataloader)
    validator.report(losses)

    assert 3 == 6
