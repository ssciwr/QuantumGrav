import QuantumGrav as QG
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pytest
from jsonschema import ValidationError


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(
        x[0], torch.cat([data.y, data.y, data.y], dim=1).to(torch.float32)
    )
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
def accuracy_config():
    return {
        "metrics": "torch.nn.HuberLoss",
        "metric_args": [],
        "metric_kwargs": {"reduction": "sum", "delta": 2.0},
    }


@pytest.fixture(scope="session")
def accuracy_config_broken():
    return {
        "metrics_broken": "foo",
        "metric_args": [],
        "metric_kwargs": {"reduction": "sum", "delta": 2.0},
    }


@pytest.fixture
def validator_object():
    device = torch.device("cpu")

    f1eval_macro = QG.evaluate.F1ScoreEval(
        average="macro",
    )
    f1eval_micro = QG.evaluate.F1ScoreEval(
        average="micro",
    )

    acceval = QG.evaluate.AccuracyEval(
        metric=torch.nn.functional.l1_loss,
        metric_args=[],
        metric_kwargs={},
    )

    compute_per_task = {
        0: {
            "f1_macro_task_0": f1eval_macro,
            "f1_micro_task_0": f1eval_micro,
            "accuracy_task_0": acceval,
        },
        1: {
            "accuracy_task_1": acceval,
            "l1_task_1": torch.nn.functional.l1_loss,
        },
    }

    get_target_per_task = {
        0: lambda x, i: torch.cat(
            [x, x, x], dim=1
        ),  # this is a bit of a hack to align the output
        1: lambda x, i: x[:, 1],
    }

    def apply_model(model, data):
        out = model(data.x, data.edge_index, data.batch)
        out[0] = (torch.sigmoid(out[0]) > 0.5).to(torch.float)
        return out

    validator = QG.DefaultValidator(
        device=device,
        criterion=compute_loss,
        compute_per_task=compute_per_task,
        get_target_per_task=get_target_per_task,
        apply_model=apply_model,
    )

    return validator


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


def test_f1eval_call(f1eval_config):
    evaluator = QG.evaluate.F1ScoreEval.from_config(f1eval_config)

    predictions = torch.tensor([[0, 1], [1, 1]])
    targets = torch.tensor([[1, 0], [0, 1]])

    # test with 2d Data
    data = {
        "target_1": [targets, targets, targets],
        "output_1": [predictions, predictions, predictions],
    }
    f1 = evaluator(data, 1)
    assert f1 == pytest.approx(1.0 / 3.0, rel=1e-3)

    with pytest.raises(KeyError, match="Task 3 not found in data for F1ScoreEval"):
        evaluator(data, 3)

    # test with 1d data
    predictions = torch.tensor([0, 1, 1, 0])
    targets = torch.tensor([1, 0, 1, 0])
    data = {
        "target_1": [targets, targets, targets],
        "output_1": [predictions, predictions, predictions],
    }
    f1 = evaluator(data, 1)
    assert f1 == pytest.approx(0.5, rel=1e-3)


def test_f1eval_call_raises(f1eval_config):
    evaluator = QG.evaluate.F1ScoreEval.from_config(f1eval_config)
    targets = torch.tensor([1, 0, 1, 0])
    predictions = torch.tensor([[0, 1], [1, 1]])
    data = {
        "target_1": [targets, targets, targets],
        "output_1": [predictions, predictions, predictions],
    }

    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator(data, 1)

    with pytest.raises(KeyError, match="Task 3 not found in data for F1ScoreEval"):
        evaluator(data, 3)


def test_accuracy_eval_creation():
    acceval = QG.evaluate.AccuracyEval(
        metric=torch.nn.HuberLoss,
        metric_args=[],
        metric_kwargs={"reduction": "sum", "delta": 2.0},
    )
    assert acceval.metric(torch.rand(5), torch.rand(5)).item() > 0.0
    assert issubclass(type(acceval.metric), torch.nn.Module)
    assert acceval.metric.reduction == "sum"

    acceval_func = QG.evaluate.AccuracyEval(
        metric=torch.nn.functional.mse_loss,
        metric_args=[],
        metric_kwargs={},
    )
    assert acceval_func.metric == torch.nn.functional.mse_loss
    assert issubclass(type(acceval_func.metric), torch.nn.Module) is False
    assert acceval_func.metric(torch.rand(5), torch.rand(5)).item() > 0.0


def test_accuracy_eval_fromconfig(accuracy_config):
    acceval = QG.evaluate.AccuracyEval.from_config(accuracy_config)
    assert issubclass(type(acceval.metric), torch.nn.Module)
    assert acceval.metric.reduction == "sum"
    assert acceval.metric(torch.rand(5), torch.rand(5)).item() > 0.0


def test_accuracy_eval_fromconfig_broken(accuracy_config_broken):
    with pytest.raises(ValidationError):
        QG.evaluate.AccuracyEval.from_config(accuracy_config_broken)


def test_accuracy_eval_call(accuracy_config):
    eval = QG.evaluate.AccuracyEval.from_config(accuracy_config)
    predictions = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    targets = torch.tensor([[0.75, 0.4], [0.23, 0.76]])
    data = {"target_1": [targets, targets], "output_1": [predictions, predictions]}
    accuracy = eval(data, 1)
    assert accuracy > 0.0

    with pytest.raises(KeyError, match="'Missing data for task 3 in AccuracyEval'"):
        eval(data, 3)


def test_default_validator_creation(gnn_model_eval):
    """Test the DefaultValidator class."""
    device = torch.device("cpu")

    f1eval_macro = QG.evaluate.F1ScoreEval(
        average="macro",
    )
    f1eval_micro = QG.evaluate.F1ScoreEval(
        average="micro",
    )

    acceval = QG.evaluate.AccuracyEval(
        metric=torch.nn.functional.l1_loss,
        metric_args=[],
        metric_kwargs={},
    )

    compute_per_task = {
        0: {
            "f1_macro": f1eval_macro,
            "f1_micro": f1eval_micro,
            "accuracy": acceval,
        },
        1: {
            "accuracy": acceval,
            "l1": torch.nn.functional.l1_loss,
        },
    }

    get_target_per_task = {
        0: lambda i, x: x[:, 0],
        1: lambda i, x: x[:, 1],
    }

    validator = QG.DefaultValidator(
        device=device,
        criterion=compute_loss,
        compute_per_task=compute_per_task,
        get_target_per_task=get_target_per_task,
        apply_model=lambda data: gnn_model_eval(data, data.edge_index, data.batch),
    )

    assert validator.device == device
    assert validator.criterion is compute_loss
    assert validator.apply_model is not None
    assert validator.data is None
    assert validator.logger is not None
    assert validator.active_tasks == []
    assert validator.get_target_per_task == get_target_per_task
    assert validator.compute_per_task == compute_per_task


def test_default_evaluator_evaluate(make_dataset, gnn_model_eval, validator_object):
    dataloader = DataLoader(make_dataset, batch_size=4)
    validator_object.evaluate(gnn_model_eval, dataloader)
    assert len(validator_object.data) == len(dataloader)
    assert validator_object.data.columns.tolist() == [
        "loss",
        "f1_macro_task_0",
        "f1_micro_task_0",
        "accuracy_task_0",
    ]  # only task 0 metrics are computed since the dummy model outputs only one task


# def test_default_evaluator_report(caplog):
#     device = torch.device("cpu")
#     evaluator = QG.DefaultEvaluator(
#         device=device,
#         criterion=compute_loss,
#         apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
#     )
#     assert len(evaluator.data) == 0
#     losses = np.random.rand(100)
#     expected_avg = np.mean(losses)
#     expected_std = np.std(losses)

#     with caplog.at_level(logging.INFO):
#         evaluator.report(losses)
#         # Test specific content
#         assert f"Average loss: {expected_avg}" in caplog.text
#         assert f"Standard deviation: {expected_std}" in caplog.text


# def test_default_tester_creation():
#     """Test the DefaultTester class."""
#     device = torch.device("cpu")
#     tester = QG.DefaultTester(
#         device=device,
#         criterion=compute_loss,
#         apply_model=None,
#     )

#     assert tester.device == device
#     assert tester.criterion is compute_loss
#     assert tester.apply_model is None


# def test_default_tester_test(make_dataloader, gnn_model_eval):
#     dataloader = make_dataloader
#     device = torch.device("cpu")
#     tester = QG.DefaultTester(
#         device=device,
#         criterion=compute_loss,
#         apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
#     )
#     testdata = tester.test(gnn_model_eval, dataloader)
#     assert len(testdata) == len(dataloader)
#     assert torch.Tensor(testdata).dtype == torch.float32


# def test_default_tester_report(caplog):
#     device = torch.device("cpu")
#     tester = QG.DefaultTester(
#         device=device,
#         criterion=compute_loss,
#         apply_model=None,
#     )
#     losses = np.random.rand(100)
#     expected_avg = np.mean(losses)
#     expected_std = np.std(losses)

#     with caplog.at_level(logging.INFO):
#         tester.report(losses)
#         # Test specific content
#         assert f"Average loss: {expected_avg}" in caplog.text
#         assert f"Standard deviation: {expected_std}" in caplog.text


# def test_default_early_stopping_creation(input):
#     """Test the DefaultEarlyStopping class."""

#     early_stopping = QG.DefaultEarlyStopping(
#         input["patience"],
#         input["delta"],
#         input["window"],
#         metric=input["metric"],
#         smoothing=input["smoothing"],
#         grace_period=input["grace_period"],
#         init_best_score=42,
#         mode="all",
#     )

#     assert early_stopping.patience == input["patience"]
#     assert early_stopping.delta == input["delta"]
#     assert early_stopping.window == input["window"]
#     assert early_stopping.best_score == [42, 42]
#     assert early_stopping.current_patience == input["patience"]
#     assert early_stopping.metric == ["loss", "other_loss"]
#     assert early_stopping.smoothing is False
#     assert early_stopping.found_better == [False, False]
#     assert early_stopping.logger is not None
#     assert early_stopping.mode == "all"

#     early_stopping_callable = QG.DefaultEarlyStopping(
#         input["patience"],
#         input["delta"],
#         input["window"],
#         metric=input["metric"],
#         smoothing=input["smoothing"],
#         grace_period=input["grace_period"],
#         init_best_score=42,
#         mode=lambda b: all(b),
#     )

#     assert callable(early_stopping_callable.mode)
#     with pytest.raises(
#         ValueError, match="Inconsistent lengths for early stopping parameters"
#     ):
#         early_stopping_callable = QG.DefaultEarlyStopping(
#             input["patience"],
#             input["delta"],
#             [12, 7, 9],
#             metric=input["metric"],
#             smoothing=input["smoothing"],
#             grace_period=input["grace_period"],
#             init_best_score=42,
#             mode=lambda b: all(b),
#         )


# @pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
# def test_default_early_stopping_check_positive(smoothed, input):
#     """Test the check method of DefaultEarlyStopping - positive."""
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=smoothed,
#     )

#     early_stopping.best_score = [12.0, 12.0]
#     losses = pd.DataFrame(
#         {
#             "loss": [
#                 0.1,
#                 0.09,
#                 0.08,
#                 0.07,
#             ],
#             "other_loss": [0.2, 0.3, 0.4, 0.5],
#         }
#     )

#     assert early_stopping(losses) is False
#     assert early_stopping.current_patience == input["patience"]
#     assert early_stopping.current_grace_period == [
#         input["grace_period"][0] - 1,
#         input["grace_period"][1] - 1,
#     ]
#     assert early_stopping.found_better == [True, True]


# @pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
# def test_default_early_stopping_check_positive_with_untracked_metric(smoothed, input):
#     """Test the check method of DefaultEarlyStopping - positive."""
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=smoothed,
#     )

#     early_stopping.best_score = [12.0, 12.0]
#     losses = pd.DataFrame(
#         {
#             "loss": [
#                 0.1,
#                 0.09,
#                 0.08,
#                 0.07,
#             ],
#             "untracked": [0.2, 0.3, 0.4, 0.5],
#         }
#     )

#     assert early_stopping(losses) is False
#     assert early_stopping.current_patience == input["patience"]
#     assert early_stopping.current_grace_period == [
#         input["grace_period"][0] - 1,
#         input["grace_period"][1] - 1,
#     ]
#     assert early_stopping.found_better == [True, True]


# @pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
# def test_default_early_stopping_check_negative(smoothed, input):
#     """Test the check method of DefaultEarlyStopping - negative."""
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=smoothed,
#     )

#     early_stopping.best_score = [1e-5, 1e-5]
#     # set grace period artificially lower
#     early_stopping.current_grace_period = [0, 0]
#     losses = pd.DataFrame(
#         {
#             "loss": [
#                 10.0,
#                 10.0,
#             ],
#             "other_loss": [
#                 10.0,
#                 10.0,
#             ],
#         }
#     )
#     assert early_stopping(losses) is False
#     assert early_stopping.current_patience == input["patience"] - 1
#     assert np.array_equal(early_stopping.current_grace_period, [0, 0])
#     assert np.array_equal(early_stopping.best_score, [1e-5, 1e-5])  # not updated


# @pytest.mark.parametrize("smoothed", [False, True], ids=["not_smoothed", "smoothed"])
# def test_default_early_stopping_triggered(smoothed, input):
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=smoothed,
#     )
#     early_stopping.best_score = [1e-4, 1e-4]
#     early_stopping.patience = 1
#     early_stopping.grace_period = [0, 0]
#     early_stopping.reset()

#     losses = pd.DataFrame(
#         {
#             "loss": [
#                 0.1,
#                 0.1,
#             ],
#             "other_loss": [
#                 0.1,
#                 0.1,
#             ],
#         }
#     )
#     assert early_stopping(losses) is True
#     assert early_stopping.current_patience == 0


# def test_default_early_stopping_add_task(input):
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=False,
#     )

#     early_stopping.add_task(1e-3, 12, "blergh", 3)

#     assert early_stopping.delta[-1] == 1e-3
#     assert early_stopping.window[-1] == 12
#     assert early_stopping.metric[-1] == "blergh"
#     assert early_stopping.grace_period[-1] == 3
#     assert early_stopping.current_grace_period[-1] == 3
#     assert early_stopping.best_score[-1] == np.inf
#     assert early_stopping.found_better == [False, False, False]


# def test_default_early_stopping_reset(input):
#     early_stopping = QG.DefaultEarlyStopping(
#         patience=input["patience"],
#         delta=input["delta"],
#         window=input["window"],
#         metric=input["metric"],
#         grace_period=input["grace_period"],
#         smoothing=False,
#     )

#     early_stopping.current_patience = input["patience"] - 2
#     early_stopping.current_grace_period = [
#         input["grace_period"][0] - 1,
#         input["grace_period"][1] - 1,
#     ]
#     early_stopping.found_better = [True, False, False]
#     early_stopping.best_score = [1, 2, 3]
#     early_stopping.reset()

#     assert early_stopping.current_patience == input["patience"]
#     assert early_stopping.current_grace_period == input["grace_period"]
#     assert early_stopping.found_better == [False] * len(input["metric"])
#     assert early_stopping.best_score == [1, 2, 3]
#     assert early_stopping.current_grace_period == early_stopping.grace_period
#     assert early_stopping.current_patience == early_stopping.patience
