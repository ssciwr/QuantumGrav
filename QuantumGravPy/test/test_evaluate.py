import logging
import QuantumGrav as QG
import numpy as np
import pytest
import pandas as pd

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def bc_transform(data: Data) -> Data:
    """A simple transform that makes the data 1D."""
    data.y = data.y[:, 0]
    return data


@pytest.fixture
def make_dataset_bc(create_data_hdf5, read_data):
    datadir, datafiles = create_data_hdf5

    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
        transform=bc_transform,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )
    return dataset


@pytest.fixture
def gnn_model_eval_bc(model_config_eval):
    """Fixture to create a GNNModel for evaluation."""
    model_config_eval["downstream_tasks"][0]["output_dim"] = 1
    model = QG.GNNModel.from_config(
        model_config_eval,
    )
    model.eval()
    return model


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    print("shape: ", x[0].shape, data.y.shape)
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
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        prefix="test",
    )

    assert evaluator.device == device
    assert evaluator.apply_model is not None


def test_f1_evaluator_evaluate(make_dataset_bc, gnn_model_eval_bc):
    "test F1Evaluator.evaluate() method"
    dataloader = DataLoader(
        make_dataset_bc,
        batch_size=4,
        shuffle=True,
        drop_last=True,  # Ensure all batches are of the same size. last batches that are bad need to be handled by hand
    )
    device = torch.device("cpu")
    model = gnn_model_eval_bc.to(device)
    evaluator = QG.F1Evaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(
            data.x, data.edge_index, data.batch
        ),  # Assuming binary classification
        prefix="test",
    )
    f1_scores = evaluator.evaluate(model, dataloader)
    f1_scores.columns == [
        "loss",
        "output",
        "target",
        "f1_per_class",
        "f1_unweighted",
        "f1_weighted",
        "f1_micro",
    ]

    # only need to check one column for the length
    assert isinstance(f1_scores, pd.DataFrame)
    assert len(f1_scores["loss"]) == len(dataloader) * 4  # batch size
    assert torch.Tensor(f1_scores["loss"]).dtype == torch.float32

    for col in [
        "f1_unweighted",
        "f1_weighted",
        "f1_micro",
    ]:
        assert f1_scores[col].min() >= 0.0
        assert f1_scores[col].max() <= 1.0


def test_f1_evaluator_report(caplog):
    "test F1Evaluator.report() method"

    device = torch.device("cpu")
    evaluator = QG.F1Evaluator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(
            data.x, data.edge_index, data.batch
        ),  # Assuming binary classification
        prefix="test",
    )

    assert len(evaluator.data) == 0
    avg_losses = np.random.rand(10)
    std_losses = np.random.rand(10)
    f1_scores_weighted = np.random.rand(10)
    f1_scores_unweighted = np.random.rand(10)
    f1_scores_micro = np.random.rand(10)
    f1_scores_per_class = np.random.rand(10, 2)

    f1_scores = pd.DataFrame(
        {
            "avg_loss": avg_losses,
            "std_loss": std_losses,
            "f1_weighted": f1_scores_weighted,
            "f1_unweighted": f1_scores_unweighted,
            "f1_micro": f1_scores_micro,
            "f1_per_class": list(f1_scores_per_class),
        }
    )

    evaluator.data = f1_scores
    assert len(evaluator.data["avg_loss"]) == 10

    with caplog.at_level(logging.INFO):
        evaluator.report(f1_scores)

        # Test specific content
        assert f"test avg loss: {f1_scores.loc[9, 'avg_loss']:.4f} +/- {f1_scores.loc[9, 'std_loss']:.4f}" in caplog.text
        assert f"test f1 score per class: {f1_scores.loc[9, 'f1_per_class']}" in caplog.text
        assert f"test f1 score unweighted: {f1_scores.loc[9, 'f1_unweighted']}" in caplog.text
        assert f"test f1 score weighted: {f1_scores.loc[9, 'f1_weighted']}" in caplog.text
        assert f"test f1 score micro: {f1_scores.loc[9, 'f1_micro']}" in caplog.text


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
