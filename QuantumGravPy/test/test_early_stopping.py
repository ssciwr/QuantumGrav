import QuantumGrav as QG
import pytest
from jsonschema import ValidationError
import copy
import pandas as pd


@pytest.fixture
def early_stoppinginput():
    return {
        "tasks": {
            0: {
                "delta": 1e-2,
                "metric": "loss",
                "grace_period": 8,
                "init_best_score": 1000000.0,
                "mode": "min",
            },
            1: {
                "delta": 1e-4,
                "metric": "other_loss",
                "grace_period": 10,
                "init_best_score": -1000000.0,
                "mode": "max",
            },
        },
        "mode": "any",
        "patience": 12,
    }


def test_default_early_stopping_creation(early_stoppinginput):
    """Test the DefaultEarlyStopping class."""

    early_stopping = QG.DefaultEarlyStopping(
        tasks=early_stoppinginput["tasks"],
        mode=early_stoppinginput["mode"],
        patience=early_stoppinginput["patience"],
    )
    for key, task in early_stopping.tasks.items():
        assert task["delta"] == early_stoppinginput["tasks"][key]["delta"]
        assert (
            task["best_score"] == early_stoppinginput["tasks"][key]["init_best_score"]
        )
        assert task["metric"] == early_stoppinginput["tasks"][key]["metric"]
        assert task["found_better"] is False
        assert task["mode"] == early_stoppinginput["tasks"][key]["mode"]
        assert early_stopping.logger is not None
    assert early_stopping.mode == "any"
    assert early_stopping.current_patience == early_stoppinginput["patience"]
    assert early_stopping.patience == early_stoppinginput["patience"]


def test_default_early_stopping_check_any_found_improvement(early_stoppinginput):
    """Test the check method of DefaultEarlyStopping - any."""
    early_stopping = QG.DefaultEarlyStopping(
        tasks=early_stoppinginput["tasks"],
        mode="all",
        patience=early_stoppinginput["patience"],
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
    assert early_stopping.current_patience == early_stoppinginput["patience"]

    for key, task in early_stopping.tasks.items():
        assert task["current_grace_period"] == task["grace_period"] - 1
        assert task["found_better"] is True  # all are better


def test_default_early_stopping_check_any_found_improvement_any(early_stoppinginput):
    """Test the check method of DefaultEarlyStopping - any."""
    early_stopping = QG.DefaultEarlyStopping(
        tasks=early_stoppinginput["tasks"],
        mode="any",
        patience=early_stoppinginput["patience"],
    )

    early_stopping.best_score = [12.0, 12.0]
    losses = pd.DataFrame(
        {
            "loss": [
                1e10,
                1e10,
                1e10,
                1e10,
            ],
            "other_loss": [0.2, 0.3, 0.4, 0.5],
        }
    )

    assert early_stopping(losses) is False
    assert early_stopping.current_patience == early_stoppinginput["patience"]

    for key, task in early_stopping.tasks.items():
        assert task["current_grace_period"] == task["grace_period"] - 1
        assert task["found_better"] == (key != 0)


def test_default_early_stopping_triggered(early_stoppinginput):
    """Test the check method of DefaultEarlyStopping - early stopping triggered."""
    early_stopping = QG.DefaultEarlyStopping(
        tasks=early_stoppinginput["tasks"],
        mode="all",
        patience=early_stoppinginput["patience"],
    )
    early_stopping.current_patience = 1
    for _, task in early_stopping.tasks.items():
        task["current_grace_period"] = 0

    early_stopping.tasks[0]["best_score"] = 1e-3
    early_stopping.tasks[1]["best_score"] = 1000

    losses = pd.DataFrame(
        {
            "loss": [
                1e10,
                1e10,
            ],
            "other_loss": [
                1e-10,
                1e-10,
            ],
        }
    )
    assert early_stopping(losses) is True
    assert early_stopping.current_patience == 0


def test_default_early_stopping_from_config(early_stoppinginput):
    """test from_config of the earlystopping class"""
    early_stopping = QG.DefaultEarlyStopping.from_config(early_stoppinginput)
    for key, task in early_stopping.tasks.items():
        assert task["delta"] == early_stoppinginput["tasks"][key]["delta"]
        assert (
            task["best_score"] == early_stoppinginput["tasks"][key]["init_best_score"]
        )
        assert task["metric"] == early_stoppinginput["tasks"][key]["metric"]
        assert task["found_better"] is False
        assert task["mode"] == early_stoppinginput["tasks"][key]["mode"]
        assert early_stopping.logger is not None
    assert early_stopping.mode == "any"
    assert early_stopping.current_patience == early_stoppinginput["patience"]
    assert early_stopping.patience == early_stoppinginput["patience"]


def test_default_early_stopping_broken_config(early_stoppinginput):
    """test from_config of the earlystopping class when it violates the json schema"""
    cfg = copy.deepcopy(early_stoppinginput)

    del cfg["tasks"][0]["delta"]  # break the config

    with pytest.raises(ValidationError, match="'delta' is a required property"):
        QG.DefaultEarlyStopping.from_config(cfg)


def test_default_early_stopping_to_config(early_stoppinginput):
    early_stopping = QG.DefaultEarlyStopping.from_config(early_stoppinginput)

    constructed_conf = early_stopping.to_config()

    assert early_stoppinginput == constructed_conf
