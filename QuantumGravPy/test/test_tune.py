import pytest
from QGTune import tune
import optuna
import yaml
import numpy as np
import time


@pytest.fixture
def get_config():
    return {
        "model": {
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": ["ref"]},
                {"in_dim": "ref", "out_dim": 64, "norm_args": ["ref"]},
            ],
            "classifier": {"in_dim": "ref"},
            "num_layers": [1, 2, 3],
            "activation": ["relu", "tanh", "sigmoid"],
            "name": "MyModel",
        },
        "training": {
            "lr": {"type": "tuple", "value": [1e-5, 1e-1, True]},
            "batch_size": [16, 32, 64],
        },
    }


@pytest.fixture
def get_base_config_file(tmp_path):
    base_config = {
        "model": {
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": [32]},
                {"in_dim": 32, "out_dim": 64, "norm_args": [64]},
            ],
            "classifier": {"in_dim": 64},
            "num_layers": 2,
            "activation": "relu",
            "name": "MyModel",
        },
        "training": {
            "lr": 0.001,
            "batch_size": 16,
        },
    }
    with open(tmp_path / "base_config.yaml", "w") as f:
        yaml.safe_dump(base_config, f)
    return tmp_path / "base_config.yaml"


@pytest.fixture
def get_best_trial():
    return {
        "model.num_layers": 2,
        "model.activation": "tanh",
        "training.lr": 0.0001,
        "training.batch_size": 32,
    }


@pytest.fixture
def get_fixed_trial(get_best_trial):
    return optuna.trial.FixedTrial(params=get_best_trial)


@pytest.fixture
def get_dependencies():
    return {
        "model": {
            "gcn_net": [
                # layer 0
                {"norm_args": ["model.gcn_net[0].out_dim"]},
                # layer 1
                {
                    "in_dim": "model.gcn_net[0].out_dim",
                    "norm_args": ["model.gcn_net[1].out_dim"],
                },
            ],
            "classifier": {"in_dim": "model.gcn_net[-1].out_dim"},
        }
    }


@pytest.fixture
def get_config_file(tmp_path, get_config):
    yaml_file = tmp_path / "sample.yaml"
    with open(yaml_file, "w") as f:
        yaml.safe_dump(get_config, f)
    return yaml_file


@pytest.fixture
def get_dependencies_file(tmp_path, get_dependencies):
    yaml_file = tmp_path / "dependencies.yaml"
    with open(yaml_file, "w") as f:
        yaml.safe_dump(get_dependencies, f)
    return yaml_file


@pytest.fixture
def get_tune_config():
    return {
        "direction": "minimize",
        "study_name": "test_study",
        "storage": None,
    }


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


def test_is_flat_list():
    assert tune._is_flat_list("something") is False
    assert tune._is_flat_list({}) is False
    assert tune._is_flat_list([]) is True
    assert tune._is_flat_list(["relu", "tanh", "sigmoid"]) is True
    assert tune._is_flat_list([{"a": 1}, {"b": 2}]) is False
    assert tune._is_flat_list([[1, 2], [3, 4]]) is False
    assert tune._is_flat_list([True, False]) is True
    assert tune._is_flat_list([None, None]) is True


def test_is_suggest_categorical():
    assert tune._is_suggest_categorical(1) is False
    assert tune._is_suggest_categorical("something") is False
    assert tune._is_suggest_categorical({}) is False
    assert tune._is_suggest_categorical([]) is False
    assert tune._is_suggest_categorical([1]) is True
    assert tune._is_suggest_categorical(["relu", "tanh", "sigmoid"]) is True
    assert tune._is_suggest_categorical([16, 32, 64]) is True
    assert tune._is_suggest_categorical([1.0, 2.0, 3.0]) is True
    assert tune._is_suggest_categorical([{"a": 1}, {"b": 2}]) is False
    assert tune._is_suggest_categorical([[1, 2], [3, 4]]) is False
    assert tune._is_suggest_categorical([True, False]) is True


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

    trial = optuna.trial.FixedTrial({"param": 1})
    assert tune._convert_to_suggestion("norm_args", [12], trial) == [12]


def test_get_suggestions_single():
    config = {"name": "MyModel"}
    trial = optuna.trial.FixedTrial({"name": "MyModel"})
    suggestions = tune.get_suggestion(config, trial, traced_param=[])
    assert suggestions == {"name": "MyModel"}

    config = {"lr": {"type": "tuple", "value": [1e-5, 1e-1, True]}}
    trial = optuna.trial.FixedTrial({"lr": 0.0001})
    suggestions = tune.get_suggestion(config, trial, traced_param=[])
    assert suggestions == {"lr": 0.0001}

    config = {"num_layers": [1, 2, 3]}
    trial = optuna.trial.FixedTrial({"num_layers": 2})
    suggestions = tune.get_suggestion(config, trial, traced_param=[])
    assert suggestions == {"num_layers": 2}

    config = {"activation": ["relu", "tanh", "sigmoid"]}
    trial = optuna.trial.FixedTrial({"activation": "tanh"})
    suggestions = tune.get_suggestion(config, trial, traced_param=[])
    assert suggestions == {"activation": "tanh"}


def test_get_suggestions_nested(get_config, get_fixed_trial):
    suggestions = tune.get_suggestion(get_config, get_fixed_trial, traced_param=[])
    assert suggestions == {
        "model": {
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": ["ref"]},
                {"in_dim": "ref", "out_dim": 64, "norm_args": ["ref"]},
            ],
            "classifier": {"in_dim": "ref"},
            "num_layers": 2,
            "activation": "tanh",
            "name": "MyModel",
        },
        "training": {
            "lr": 0.0001,
            "batch_size": 32,
        },
    }


def test_resolve_dependencies(get_config):
    ref_path = "training.lr.value"
    current = tune._resolve_dependencies(get_config, ref_path)
    assert np.isclose(current[0], 1e-5)
    assert np.isclose(current[1], 1e-1)
    assert current[2] is True

    ref_path = "model.gcn_net[0].out_dim"
    current = tune._resolve_dependencies(get_config, ref_path)
    assert current == 32

    ref_path = "model.gcn_net[-1].out_dim"
    current = tune._resolve_dependencies(get_config, ref_path)
    assert current == 64


def test_apply_dependencies(get_config, get_dependencies):
    config_with_deps = tune.apply_dependencies(get_config, get_dependencies)
    assert config_with_deps["model"]["gcn_net"][0]["norm_args"] == [32]
    assert config_with_deps["model"]["gcn_net"][1]["in_dim"] == 32
    assert config_with_deps["model"]["gcn_net"][1]["norm_args"] == [64]
    assert config_with_deps["model"]["classifier"]["in_dim"] == 64


def test_load_yaml_invalid(tmp_path):
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(None, description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml("", description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(1, description="Test")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(tmp_path / "non_existent.yaml", description="Test")


def test_load_yaml_valid(get_config_file):
    config = tune.load_yaml(get_config_file, description="Sample YAML")
    assert "model" in config
    assert "training" in config
    assert "gcn_net" in config["model"]
    assert len(config["model"]["gcn_net"]) == 2
    assert config["model"]["gcn_net"][0]["in_dim"] == 16
    assert config["model"]["gcn_net"][0]["out_dim"] == 32
    assert config["model"]["gcn_net"][1]["in_dim"] == "ref"
    assert config["training"]["lr"]["type"] == "tuple"


def test_build_search_space_with_dependencies_tune_all(
    get_config_file, get_dependencies_file, get_fixed_trial
):
    search_space = tune.build_search_space_with_dependencies(
        get_config_file,
        get_dependencies_file,
        get_fixed_trial,
        tune_model=True,
        tune_training=True,
        base_settings_file=None,
    )

    expected_search_space = {
        "model": {
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": [32]},
                {"in_dim": 32, "out_dim": 64, "norm_args": [64]},
            ],
            "classifier": {"in_dim": 64},
            "num_layers": 2,
            "activation": "tanh",
            "name": "MyModel",
        },
        "training": {
            "lr": 0.0001,
            "batch_size": 32,
        },
    }
    assert search_space == expected_search_space


def test_build_search_space_tune_model_only(
    get_config_file, get_dependencies_file, get_fixed_trial, get_base_config_file
):
    search_space = tune.build_search_space_with_dependencies(
        get_config_file,
        get_dependencies_file,
        get_fixed_trial,
        tune_model=True,
        tune_training=False,
        base_settings_file=get_base_config_file,
    )

    expected_search_space = {
        "model": {
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": [32]},
                {"in_dim": 32, "out_dim": 64, "norm_args": [64]},
            ],
            "classifier": {"in_dim": 64},
            "num_layers": 2,
            "activation": "tanh",
            "name": "MyModel",
        },
        "training": {  # obtained from base config
            "lr": 0.001,
            "batch_size": 16,
        },
    }
    assert search_space == expected_search_space


def test_build_search_space_tune_training_only(
    get_config_file, get_dependencies_file, get_fixed_trial, get_base_config_file
):
    search_space = tune.build_search_space_with_dependencies(
        get_config_file,
        get_dependencies_file,
        get_fixed_trial,
        tune_model=False,
        tune_training=True,
        base_settings_file=get_base_config_file,
    )

    expected_search_space = {
        "model": {  # obtained from base config
            "gcn_net": [
                {"in_dim": 16, "out_dim": 32, "norm_args": [32]},
                {"in_dim": 32, "out_dim": 64, "norm_args": [64]},
            ],
            "classifier": {"in_dim": 64},
            "num_layers": 2,
            "activation": "relu",
            "name": "MyModel",
        },
        "training": {
            "lr": 0.0001,
            "batch_size": 32,
        },
    }
    assert search_space == expected_search_space


def test_create_study(get_tune_config):
    study = tune.create_study(get_tune_config)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.study_name == "test_study"


def test_create_study_with_storage(get_tune_config, tmp_path):
    storage_path = tmp_path / "optuna_study.log"
    get_tune_config["storage"] = storage_path
    study = tune.create_study(get_tune_config)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert study.study_name == "test_study"
    assert study._storage is not None
    assert isinstance(study._storage, optuna.storages.JournalStorage)


def test_save_best_trial(tmp_path):
    study = optuna.create_study(direction="minimize", storage=None)
    best_trial = optuna.trial.FrozenTrial(
        number=0,
        state=optuna.trial.TrialState.COMPLETE,
        value=0.5,
        datetime_start=time.time(),
        datetime_complete=time.time(),
        params={"lr": 0.01, "n_layers": 3},
        distributions={
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
            "n_layers": optuna.distributions.IntDistribution(1, 5),
        },
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=0,
    )
    study.add_trial(best_trial)

    output_file = tmp_path / "best_trial.yaml"
    tune.save_best_trial(study, output_file)
    loaded_config = tune.load_yaml(output_file, description="Best Trial Config")
    assert loaded_config == {"lr": 0.01, "n_layers": 3}


def test_save_best_config(tmp_path):
    built_search_space = {
        "model": {
            "nn": [
                {
                    "in_dim": "waiting",
                    "out_dim": "waiting",
                    "dropout": "waiting",
                },
                {
                    "in_dim": "waiting",
                    "out_dim": "waiting",
                    "dropout": "waiting",
                },
            ]
        }
    }
    best_trial = {
        "model.nn.0.in_dim": 784,
        "model.nn.0.out_dim": 32,
        "model.nn.0.dropout": 0.5,
        "model.nn.1.in_dim": 32,
        "model.nn.1.out_dim": 32,
        "model.nn.1.dropout": 0.1,
    }
    depmap = {
        "model": {
            "nn": [
                {},
                {
                    "in_dim": "model.nn[0].out_dim",
                },
            ]
        }
    }
    built_search_space_file = tmp_path / "built_search_space.yaml"
    with open(built_search_space_file, "w") as f:
        yaml.safe_dump(built_search_space, f)
    best_trial_file = tmp_path / "best_trial.yaml"
    with open(best_trial_file, "w") as f:
        yaml.safe_dump(best_trial, f)
    depmap_file = tmp_path / "depmap.yaml"
    with open(depmap_file, "w") as f:
        yaml.safe_dump(depmap, f)
    output_file = tmp_path / "best_config.yaml"

    tune.save_best_config(
        built_search_space_file,
        best_trial_file,
        depmap_file,
        output_file,
    )

    with open(output_file, "r") as f:
        best_config = yaml.safe_load(f)

    expected_config = {
        "model": {
            "nn": [
                {
                    "in_dim": 784,
                    "out_dim": 32,
                    "dropout": 0.5,
                },
                {
                    "in_dim": 32,
                    "out_dim": 32,
                    "dropout": 0.1,
                },
            ]
        }
    }
    assert best_config == expected_config
