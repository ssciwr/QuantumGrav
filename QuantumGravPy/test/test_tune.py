import pytest
from QGTune import tune
import optuna
import yaml


@pytest.fixture
def sample_yaml_file(tmp_path):
    yaml_content = """
model:
  lr:
    type: tuple
    value: [1e-5, 1e-1, true]
  num_layers: [1,2,3]
  activation: ['relu', 'tanh', 'sigmoid']
  name: 'MyModel'
"""
    yaml_file = tmp_path / "sample.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


def test_is_yaml_tuple_of_3():
    assert tune._is_yaml_tuple_of_3(1) is False
    assert tune._is_yaml_tuple_of_3("something") is False
    assert tune._is_yaml_tuple_of_3({}) is False
    assert tune._is_yaml_tuple_of_3({"type": "tuple"}) is False
    assert tune._is_yaml_tuple_of_3({"type": "tuple", "value": []}) is False
    assert tune._is_yaml_tuple_of_3({"type": "tuple", "value": [1]}) is False
    assert tune._is_yaml_tuple_of_3({"type": "list", "value": [1, 2, 3]}) is False
    assert tune._is_yaml_tuple_of_3({"type": "tuple", "value": [1, 2, 3]}) is True
    assert tune._is_yaml_tuple_of_3({"type": "tuple", "value": [1, 2, 3, 4]}) is False


def test_is_suggest_categorical():
    assert tune._is_suggest_categorical(1) is False
    assert tune._is_suggest_categorical("something") is False
    assert tune._is_suggest_categorical({}) is False
    assert tune._is_suggest_categorical([]) is False
    assert tune._is_suggest_categorical([1]) is True
    assert tune._is_suggest_categorical(["relu", "tanh", "sigmoid"]) is True
    assert tune._is_suggest_categorical([16, 32, 64]) is True


def test_is_suggest_float():
    assert tune._is_suggest_float(1) is False
    assert tune._is_suggest_float("something") is False
    assert tune._is_suggest_float({"type": "tuple", "value": [1.0, 2.0, 0.1]}) is True
    assert (
        tune._is_suggest_float({"type": "tuple", "value": [0.001, 0.1, 0.001]}) is True
    )
    assert (
        tune._is_suggest_float({"type": "tuple", "value": [1e-5, 1e-1, True]}) is True
    )
    assert tune._is_suggest_float({"type": "tuple", "value": [1.0, 2.0, False]}) is True
    assert (
        tune._is_suggest_float({"type": "tuple", "value": [1.0, 2.0, "something"]})
        is False
    )
    assert tune._is_suggest_float({"type": "tuple", "value": [1, 2, 0.1]}) is False
    assert tune._is_suggest_float({"type": "tuple", "value": [1, 2, 1]}) is False
    assert tune._is_suggest_float({"type": "tuple", "value": ["a", "b", "c"]}) is False


def test_is_suggest_int():
    assert tune._is_suggest_int(1) is False
    assert tune._is_suggest_int("something") is False
    assert tune._is_suggest_int({"type": "tuple", "value": [1, 2, 1]}) is True
    assert tune._is_suggest_int({"type": "tuple", "value": [16, 64, 2]}) is True
    assert tune._is_suggest_int({"type": "tuple", "value": [1.0, 100, 10]}) is False
    assert tune._is_suggest_int({"type": "tuple", "value": [1, 2, -1]}) is True
    assert tune._is_suggest_int({"type": "tuple", "value": [1, 2.0, 1]}) is False
    assert tune._is_suggest_int({"type": "tuple", "value": [1, 2, 0.1]}) is False
    assert tune._is_suggest_int({"type": "tuple", "value": ["a", "b", "c"]}) is False


def test_convert_to_suggestion():
    trial = optuna.trial.FixedTrial({"param": 1})
    assert tune._convert_to_suggestion("param", 1, trial) == 1

    trial = optuna.trial.FixedTrial({"param": "something"})
    assert tune._convert_to_suggestion("param", "something", trial) == "something"

    trial = optuna.trial.FixedTrial({"param": 1.5})
    assert tune._convert_to_suggestion(
        "param", {"type": "tuple", "value": [1.0, 2.0, 0.1]}, trial
    ) == trial.suggest_float("param", 1.0, 2.0, step=0.1)

    trial = optuna.trial.FixedTrial({"param": 0.0001})
    assert tune._convert_to_suggestion(
        "param", {"type": "tuple", "value": [1e-5, 1e-1, True]}, trial
    ) == trial.suggest_float("param", 1e-5, 1e-1, log=True)

    trial = optuna.trial.FixedTrial({"param": 50})
    assert tune._convert_to_suggestion(
        "param", {"type": "tuple", "value": [16, 64, 2]}, trial
    ) == trial.suggest_int("param", 16, 64, step=2)

    trial = optuna.trial.FixedTrial({"param": "tanh"})
    assert tune._convert_to_suggestion(
        "param", ["relu", "tanh", "sigmoid"], trial
    ) == trial.suggest_categorical("param", ["relu", "tanh", "sigmoid"])


def test_get_suggestions():
    config = {"name": "MyModel"}
    trial = optuna.trial.FixedTrial({"param": "MyModel"})
    suggestions = tune.get_suggestion(config, trial)
    assert suggestions == {"name": "MyModel"}

    config = {"lr": {"type": "tuple", "value": [1e-5, 1e-1, True]}}
    trial = optuna.trial.FixedTrial({"lr": 0.0001})
    suggestions = tune.get_suggestion(config, trial)
    assert suggestions == {"lr": 0.0001}

    config = {"num_layers": [1, 2, 3]}
    trial = optuna.trial.FixedTrial({"num_layers": 2})
    suggestions = tune.get_suggestion(config, trial)
    assert suggestions == {"num_layers": 2}

    config = {"activation": ["relu", "tanh", "sigmoid"]}
    trial = optuna.trial.FixedTrial({"activation": "tanh"})
    suggestions = tune.get_suggestion(config, trial)
    assert suggestions == {"activation": "tanh"}


def test_load_yaml_invalid(tmp_path):
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(None, description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml("", description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(1, description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(tmp_path / "non_existent.yaml", description="Test")


def test_load_yaml_valid(sample_yaml_file):
    config = tune.load_yaml(sample_yaml_file, description="Sample YAML")
    assert "model" in config
    assert "lr" in config["model"]
    assert config["model"]["lr"]["type"] == "tuple"
    assert config["model"]["lr"]["value"] == ["1e-5", "1e-1", True]
    assert "num_layers" in config["model"]
    assert config["model"]["num_layers"] == [1, 2, 3]
    assert "activation" in config["model"]
    assert config["model"]["activation"] == ["relu", "tanh", "sigmoid"]
    assert "name" in config["model"]
    assert config["model"]["name"] == "MyModel"
