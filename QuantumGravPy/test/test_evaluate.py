import QuantumGrav as QG
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import logging
import pytest
from jsonschema import ValidationError
import copy


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(
        x[0], torch.cat([data.y, data.y, data.y], dim=1).to(torch.float32)
    )
    return loss


def get_target_0(tgt: torch.Tensor, _: int) -> torch.Tensor:
    return torch.cat([tgt, tgt, tgt], dim=1)


def get_target_1(tgt: torch.Tensor, _: int) -> torch.Tensor:
    return tgt[:, 1]


def apply_model(model, data):
    out = model(data.x, data.edge_index, data.batch)
    out[0] = (torch.sigmoid(out[0]) > 0.5).to(torch.float)
    return out


@pytest.fixture(scope="session")
def accuracy_config():
    return {
        "metric": "torch.nn.HuberLoss",
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

    validator = QG.DefaultValidator(
        device=device,
        criterion=compute_loss,
        compute_per_task=compute_per_task,
        get_target_per_task=get_target_per_task,
        apply_model=apply_model,
    )

    return validator


@pytest.fixture(scope="session")
def validator_config():
    QG.utils.register_evaluation_function("compute_loss", compute_loss)
    QG.utils.register_evaluation_function("get_target_0", get_target_0)
    QG.utils.register_evaluation_function("get_target_1", get_target_1)
    QG.utils.register_evaluation_function("apply_model", apply_model)

    return {
        "device": "cpu",
        "criterion": {
            "name": "compute_loss",
            "args": [],
            "kwargs": {},
        },
        "compute_per_task": {
            "0": [
                {
                    "name": "QuantumGrav.evaluate.F1ScoreEval",
                    "name_in_data": "f1_macro",
                    "average": "macro",
                },
                {
                    "name": "QuantumGrav.evaluate.F1ScoreEval",
                    "name_in_data": "f1_micro",
                    "average": "micro",
                },
                {
                    "name": "QuantumGrav.evaluate.AccuracyEval",
                    "metric": "torch.nn.functional.l1_loss",
                    "name_in_data": "acc",
                    "metric_args": [],
                    "metric_kwargs": {},
                },
            ],
            "1": [
                {
                    "name": "QuantumGrav.evaluate.AccuracyEval",
                    "metric": "torch.nn.functional.l1_loss",
                    "name_in_data": "acc",
                    "kwargs": {
                        "metric_args": [],
                        "metric_kwargs": {},
                    },
                },
            ],
        },
        "get_target_per_task": {"0": "get_target_0", "1": "get_target_1"},
        "apply_model": {"name": "apply_model", "args": [], "kwargs": {}},
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

    acceval_none = QG.evaluate.AccuracyEval(
        metric=None, metric_args=[], metric_kwargs={}
    )
    assert issubclass(type(acceval_none.metric), torch.nn.Module) is True
    assert acceval_func.metric(torch.rand(5), torch.rand(5)).item() > 0.0
    assert acceval.metric.reduction == "sum"


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
        apply_model=apply_model,
    )

    assert validator.device == device
    assert validator.criterion is compute_loss
    assert validator.apply_model is apply_model
    assert validator.data is None
    assert validator.logger is not None
    assert validator.active_tasks == []
    assert validator.get_target_per_task == get_target_per_task
    assert validator.compute_per_task == compute_per_task


def test_default_validator_evaluate(make_dataset, gnn_model_eval, validator_object):
    dataloader = DataLoader(make_dataset, batch_size=4)
    validator_object.evaluate(gnn_model_eval, dataloader)
    assert len(validator_object.data) == len(dataloader)
    assert validator_object.data.columns.tolist() == [
        "loss",
        "f1_macro_task_0",
        "f1_micro_task_0",
        "accuracy_task_0",
    ]  # only task 0 metrics are computed since the dummy model outputs only one task


def test_default_validator_report(
    make_dataset, gnn_model_eval, validator_object, caplog
):
    dataloader = DataLoader(make_dataset, batch_size=4)
    validator_object.evaluate(gnn_model_eval, dataloader)

    with caplog.at_level(logging.INFO):
        validator_object.report()
    assert "Validation Results:" in caplog.text
    for name in [
        "loss",
        "f1_macro_task_0",
        "f1_micro_task_0",
        "accuracy_task_0",
    ]:
        assert name in caplog.text


def test_default_validator_from_config(make_dataset, gnn_model_eval, validator_config):
    validator = QG.DefaultValidator.from_config(validator_config)
    device = torch.device("cpu")
    assert validator.apply_model is apply_model
    assert callable(validator.apply_model)
    assert validator.device == device
    assert callable(validator.criterion)
    assert validator.data is None
    assert validator.logger is not None
    assert validator.active_tasks == []
    assert len(validator.get_target_per_task) == 2
    assert len(validator.compute_per_task) == 2
    assert len(validator.compute_per_task[0]) == 3
    assert len(validator.compute_per_task[1]) == 1

    # run it
    dataloader = DataLoader(make_dataset, batch_size=4)
    validator.evaluate(gnn_model_eval, dataloader)
    assert len(validator.data) == len(dataloader)
    assert validator.data.columns.tolist() == [
        "loss",
        "f1_macro",
        "f1_micro",
        "acc",
    ]  # only task 0 metrics are computed since the dummy model outputs only one task


def test_default_validator_from_config_throws(validator_config):
    # 1) Invalid top-level schema: remove a required field -> ValueError("Invalid configuration")
    cfg = copy.deepcopy(validator_config)
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad.pop("criterion", None)
    with pytest.raises(ValueError, match="Invalid configuration"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 2) Criterion cannot be imported -> ValueError("Failed to import criterion")
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["criterion"]["name"] = "does.not.exist"
    with pytest.raises(ValueError, match="Failed to import criterion"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 3) Missing name_in_data in a monitor entry -> KeyError for 'name_in_data'
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["compute_per_task"]["0"][0] = {
        "name": "QuantumGrav.evaluate.F1ScoreEval",
        # intentionally omit name_in_data
        "average": "macro",
    }
    with pytest.raises(KeyError, match="name_in_data"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 4) Monitor type import fails -> ValueError from _find_function_or_class
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["compute_per_task"]["0"][0] = {
        "name": "totally.not.there",
        "name_in_data": "broken_metric",
    }
    with pytest.raises(ValueError, match="Failed to import|import"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 5) Target getter cannot be imported -> ValueError
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["get_target_per_task"]["1"] = "totally.not.there"
    with pytest.raises(ValueError, match="Failed to import|import"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 6) Target getter is not callable (use a non-callable attribute)
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["get_target_per_task"]["0"] = "numpy.pi"
    with pytest.raises(ValueError, match="not callable"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 7) apply_model import fails -> ValueError
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["apply_model"]["name"] = "nowhere.func"
    with pytest.raises(ValueError, match="Failed to import apply_model|import"):
        QG.DefaultValidator.from_config(cfg_bad)

    # 8) apply_model resolves to a non-callable/non-class -> ValueError
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["apply_model"]["name"] = "numpy.pi"
    with pytest.raises(ValueError, match="apply_model type is not a class or callable"):
        QG.DefaultValidator.from_config(cfg_bad)
