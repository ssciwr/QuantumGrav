import logging
import QuantumGrav as QG
from torch_geometric.data import Data
import torch
import torch_geometric
import pytest
from functools import partial


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0].unsqueeze(0), data.y.to(torch.float32))
    return loss


class DummyMonitor1:
    def __init__(self, offset: float = 0.0):
        self.offset = offset
        self.loss = torch.nn.MSELoss()

    def __call__(self, predictions, targets):
        return self.loss(torch.cat(predictions), torch.cat(targets)) + self.offset  #


def dummymonitor_2(predictions, targets):
    return torch.tensor(3.14)


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
        "aggregate_pooling_type": partial(torch.cat, dim=1),
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


@pytest.fixture
def tasks():
    return [
        {
            "name": "first",
            "monitor": DummyMonitor1,
            "args": [],
            "kwargs": {"offset": 0.2},
        },
        {
            "name": "second",
            "monitor": dummymonitor_2,
        },
        {
            "name": "third",
            "monitor": DummyMonitor1,
            "args": [],
            "kwargs": {"offset": 0.1},
        },
        {
            "name": "fourth",
            "monitor": dummymonitor_2,
        },
    ]


@pytest.fixture
def config(tasks):
    config = {
        "device": "cpu",
        "criterion": compute_loss,
        "evaluator_tasks": tasks,
        "apply_model": lambda data: gnn_model_eval(data, data.edge_index, data.batch),
    }

    return config


def test_default_validator_creation(gnn_model_eval, tasks):
    """Test the Validator class."""
    device = torch.device("cpu")
    evaluator = QG.Validator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda data: gnn_model_eval(data, data.edge_index, data.batch),
        evaluator_tasks=tasks,
    )

    assert evaluator.device == device
    assert evaluator.criterion is compute_loss
    assert evaluator.apply_model is not None
    assert set(evaluator.data.columns) == {
        "loss_avg",
        "loss_max",
        "loss_min",
        "first",
        "second",
        "third",
        "fourth",
    }

    assert len(evaluator.tasks) == 4
    assert evaluator.tasks[0][0] == "first"

    assert isinstance(evaluator.tasks[0][1], DummyMonitor1)
    assert evaluator.tasks[1] == ("second", dummymonitor_2)

    assert evaluator.tasks[2][0] == "third"
    assert isinstance(evaluator.tasks[2][1], DummyMonitor1)
    assert evaluator.tasks[3] == ("fourth", dummymonitor_2)


def test_default_validator_fromconfig(gnn_model_eval, config):
    evaluator = QG.Validator.from_config(config)
    device = torch.device("cpu")
    assert evaluator.device == device
    assert evaluator.criterion is compute_loss
    assert evaluator.apply_model is not None
    assert set(evaluator.data.columns) == {
        "loss_avg",
        "loss_max",
        "loss_min",
        "first",
        "second",
        "third",
        "fourth",
    }
    assert len(evaluator.tasks) == 4
    assert evaluator.tasks[0][0] == "first"

    assert isinstance(evaluator.tasks[0][1], DummyMonitor1)
    assert evaluator.tasks[1] == ("second", dummymonitor_2)

    assert evaluator.tasks[2][0] == "third"
    assert isinstance(evaluator.tasks[2][1], DummyMonitor1)
    assert evaluator.tasks[3] == ("fourth", dummymonitor_2)


def test_default_validator_evaluate(make_dataloader, gnn_model_eval, tasks):
    dataloader = make_dataloader
    device = torch.device("cpu")
    model = gnn_model_eval.to(device)
    evaluator = QG.Validator(
        device=device,
        criterion=compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
        evaluator_tasks=tasks,
    )

    assert len(evaluator.data) == 0
    current_data = evaluator.evaluate(model, dataloader)
    assert len(evaluator.data) == 1
    assert len(current_data) == 1
    assert set(current_data.columns) == {
        "first",
        "second",
        "third",
        "fourth",
        "loss_avg",
        "loss_min",
        "loss_max",
    }


def test_default_validator_report(caplog, make_dataloader, gnn_model_eval, tasks):
    device = torch.device("cpu")
    dataloader = make_dataloader
    model = gnn_model_eval.to(device)
    evaluator = QG.Validator(
        device=device,
        criterion=compute_loss,
        evaluator_tasks=tasks,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    assert len(evaluator.data) == 0
    current_data = evaluator.evaluate(model, dataloader)

    with caplog.at_level(logging.INFO):
        evaluator.report(current_data)
        # Test specific content
        assert "Validation results:" in caplog.text
        assert "\n%s", (
            current_data.tail(1).to_string(float_format="{:.4f}".format) in caplog.text
        )


def test_default_test_report(caplog, make_dataloader, gnn_model_eval, tasks):
    device = torch.device("cpu")
    dataloader = make_dataloader
    model = gnn_model_eval.to(device)
    evaluator = QG.Tester(
        device=device,
        criterion=compute_loss,
        evaluator_tasks=tasks,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch)[0],
    )
    assert len(evaluator.data) == 0
    current_data = evaluator.evaluate(model, dataloader)

    with caplog.at_level(logging.INFO):
        evaluator.report(current_data)
        # Test specific content
        assert "Testing results:" in caplog.text
        assert "\n%s", (
            current_data.tail(1).to_string(float_format="{:.4f}".format) in caplog.text
        )
