import pytest
from QGTune import tune
from QuantumGrav import config_utils as cfg
import optuna
import yaml
import numpy as np
import time
from QuantumGrav.gnn_block import GNNBlock
from torch_geometric.nn.conv.sage_conv import SAGEConv
import copy


@pytest.fixture
def get_config(yaml_text):
    loader = cfg.get_loader()
    config = yaml.load(yaml_text, Loader=loader)
    return config


@pytest.fixture
def get_best_trial():
    return {
        "model.layers": 2,
        "model.lr": 0.01,
        "model.foo.1.x": 1,
        "trainer.epochs": 5,
        "trainer.lr": 0.001,
        "trainer.drop_rate": 0.3,
    }


@pytest.fixture
def get_suggestions_with_best_trial():
    return {
        "model": {
            "name": "test_model",
            "layers": 2,
            "type": GNNBlock,
            "convtype": SAGEConv,
            "bs": {
                "type": "coupled-sweep-mapping",
                "target": ["model", "layers"],
            },
            "lr": 0.01,
            "foo": [
                {"x": 3, "y": 5},
                {"x": 1, "y": 2},
            ],
            "bar": [
                {
                    "x": {
                        "type": "coupled-sweep-mapping",
                        "target": ["model", "foo", 1, "x"],
                    }
                }
            ],
            "baz": [
                {
                    "x": {
                        "type": "coupled-sweep-mapping",
                        "target": ["model", "foo", 1, "x"],
                    }
                }
            ],
        },
        "trainer": {
            "epochs": 5,
            "lr": 0.001,
            "drop_rate": 0.3,
            "foo_ref": {
                "type": "reference",
                "target": ["model", "foo", 1, "x"],
            },
        },
    }


@pytest.fixture
def get_coupled_sweep_mapping():
    return {
        "model.bs": {
            1: 16,
            2: 32,
        },
        "model.bar.0.x": {
            1: -1,
            2: -2,
        },
        "model.baz.0.x": {
            1: -10,
            2: -20,
        },
    }


@pytest.fixture
def get_fixed_trial(get_best_trial):
    return optuna.trial.FixedTrial(params=get_best_trial)


@pytest.fixture
def get_config_file(tmp_path, get_config):
    yaml_file = tmp_path / "sample.yaml"
    with open(yaml_file, "w") as f:
        yaml.safe_dump(get_config, f)
    return yaml_file


@pytest.fixture
def get_tune_config():
    return {
        "direction": "minimize",
        "study_name": "test_study",
        "storage": None,
    }


def test_is_flat_list():
    assert tune.is_flat_list("something") is False
    assert tune.is_flat_list({}) is False
    assert tune.is_flat_list([]) is True
    assert tune.is_flat_list(["relu", "tanh", "sigmoid"]) is True
    assert tune.is_flat_list([{"a": 1}, {"b": 2}]) is False
    assert tune.is_flat_list([[1, 2], [3, 4]]) is False
    assert tune.is_flat_list([True, False]) is True
    assert tune.is_flat_list([None, None]) is True


def test_is_categorical_suggestion():
    assert tune.is_categorical_suggestion(1) is False
    assert tune.is_categorical_suggestion("something") is False
    assert tune.is_categorical_suggestion({}) is False
    assert tune.is_categorical_suggestion([]) is False
    assert tune.is_categorical_suggestion([1]) is True
    assert tune.is_categorical_suggestion(["relu", "tanh", "sigmoid"]) is True
    assert tune.is_categorical_suggestion([16, 32, 64]) is True
    assert tune.is_categorical_suggestion([1.0, 2.0, 3.0]) is True
    assert tune.is_categorical_suggestion([{"a": 1}, {"b": 2}]) is False
    assert tune.is_categorical_suggestion([[1, 2], [3, 4]]) is False
    assert tune.is_categorical_suggestion([True, False]) is True


def testis_float_suggestion():
    assert tune.is_float_suggestion(1) is False
    assert tune.is_float_suggestion("something") is False
    assert tune.is_float_suggestion((1.0, 2.0, 0.1)) is True
    assert tune.is_float_suggestion((1e-5, 1e-1, True)) is True
    assert tune.is_float_suggestion((1.0, 2.0, "something")) is False
    assert tune.is_float_suggestion((1, 2, 0.1)) is False
    assert tune.is_float_suggestion((1, 2, 1)) is False
    assert tune.is_float_suggestion(("a", "b", "c")) is False


def test_is_int_suggestion():
    assert tune.is_int_suggestion(1) is False
    assert tune.is_int_suggestion("something") is False
    assert tune.is_int_suggestion((1, 2, 1)) is True
    assert tune.is_int_suggestion((1.0, 100, 10)) is False
    assert tune.is_int_suggestion((1, 2, -1)) is True
    assert tune.is_int_suggestion(("a", "b", "c")) is False


def test_get_value_of_ref(get_config):
    ref_path = ["model", "foo", 0, "x"]
    current = tune.get_value_of_ref(get_config, ref_path)
    assert current == 3

    ref_path = ["trainer", "lr", "type"]
    current = tune.get_value_of_ref(get_config, ref_path)
    assert current == "range"

    ref_path = ["model", "baz", -1, "x", "type"]
    current = tune.get_value_of_ref(get_config, ref_path)
    assert current == "coupled-sweep"


def test_get_value_of_ref_invalid(get_config):
    ref_path = ["model", "non_existent", 0, "x"]
    with pytest.raises(ValueError):
        tune.get_value_of_ref(get_config, ref_path)


def test_convert_to_suggestion(get_config):
    # non-tuned values
    trial = optuna.trial.FixedTrial({"param": 1})
    assert tune.convert_to_suggestion("param", 1, trial, get_config) == 1

    trial = optuna.trial.FixedTrial({"param": "something"})
    assert (
        tune.convert_to_suggestion("param", "something", trial, get_config)
        == "something"
    )

    trial = optuna.trial.FixedTrial({"param": 1})
    assert tune.convert_to_suggestion("norm_args", [12], trial, get_config) == [12]

    # range nodes
    trial = optuna.trial.FixedTrial({"param": 1.5})
    assert tune.convert_to_suggestion(
        "param", {"type": "range", "tune_values": (1.0, 2.0, 0.1)}, trial, get_config
    ) == trial.suggest_float("param", 1.0, 2.0, step=0.1)

    trial = optuna.trial.FixedTrial({"param": 0.0001})
    assert tune.convert_to_suggestion(
        "param", {"type": "range", "tune_values": (1e-5, 1e-1, True)}, trial, get_config
    ) == trial.suggest_float("param", 1e-5, 1e-1, log=True)

    trial = optuna.trial.FixedTrial({"param": 50})
    assert tune.convert_to_suggestion(
        "param", {"type": "range", "tune_values": (16, 64, 2)}, trial, get_config
    ) == trial.suggest_int("param", 16, 64, step=2)

    # sweep nodes
    trial = optuna.trial.FixedTrial({"param": "tanh"})
    assert tune.convert_to_suggestion(
        "param",
        {"type": "sweep", "values": ["relu", "tanh", "sigmoid"]},
        trial,
        get_config,
    ) == trial.suggest_categorical("param", ["relu", "tanh", "sigmoid"])

    # coupled-sweep nodes
    trial = optuna.trial.FixedTrial({"param": 1})
    current_node = get_config["model"]["bar"][0]["x"]
    assert tune.convert_to_suggestion(
        "param",
        current_node,
        trial,
        get_config,
    ) == {
        "type": "coupled-sweep-mapping",
        "target": current_node.get("target"),
        "mapping": {
            1: -1,
            2: -2,
        },
    }

    # coupled-sweep nodes with raised error
    trial = optuna.trial.FixedTrial({"param": 1})
    non_sweep_node = {
        "type": "coupled-sweep",
        "target": ["trainer", "epochs"],
        "values": [10, 20, 30],
    }
    with pytest.raises(ValueError):
        tune.convert_to_suggestion(
            "param",
            non_sweep_node,
            trial,
            get_config,
        )

    invalid_length_node = {
        "type": "coupled-sweep",
        "target": ["model", "foo", 1, "x"],
        "values": [10],
    }
    with pytest.raises(ValueError):
        tune.convert_to_suggestion(
            "param",
            invalid_length_node,
            trial,
            get_config,
        )


def test_get_suggestions_single():
    config = {"name": "MyModel"}
    trial = optuna.trial.FixedTrial({"name": "MyModel"})
    suggestions, _ = tune.get_suggestion(
        config, config, trial, traced_param=[], coupled_sweep_mapping={}
    )
    assert suggestions == {"name": "MyModel"}

    config = {"lr": {"type": "range", "tune_values": (1e-5, 1e-1, True)}}
    trial = optuna.trial.FixedTrial({"lr": 0.0001})
    suggestions, _ = tune.get_suggestion(
        config, config, trial, traced_param=[], coupled_sweep_mapping={}
    )
    assert suggestions == {"lr": 0.0001}

    config = {"num_layers": {"type": "range", "tune_values": (1, 3, 1)}}
    trial = optuna.trial.FixedTrial({"num_layers": 2})
    suggestions, _ = tune.get_suggestion(
        config, config, trial, traced_param=[], coupled_sweep_mapping={}
    )
    assert suggestions == {"num_layers": 2}

    config = {"activation": {"type": "sweep", "values": ["relu", "tanh", "sigmoid"]}}
    trial = optuna.trial.FixedTrial({"activation": "tanh"})
    suggestions, _ = tune.get_suggestion(
        config, config, trial, traced_param=[], coupled_sweep_mapping={}
    )
    assert suggestions == {"activation": "tanh"}


@pytest.mark.filterwarnings("ignore::UserWarning")  # Optuna warning about (1, 6, 2)
def test_get_suggestions_nested(
    get_config,
    get_fixed_trial,
    get_suggestions_with_best_trial,
    get_coupled_sweep_mapping,
):
    suggestions, coupled_sweep_mapping = tune.get_suggestion(
        config=get_config,
        current_node=get_config,
        trial=get_fixed_trial,
        traced_param=[],
    )
    assert suggestions == get_suggestions_with_best_trial
    assert coupled_sweep_mapping == get_coupled_sweep_mapping


def test_resolve_references_one_node(
    get_suggestions_with_best_trial, get_coupled_sweep_mapping
):
    # change only one node
    config = copy.deepcopy(get_suggestions_with_best_trial)
    current_node = config.get("model").get("bs")
    tune.resolve_references(
        config=config,
        node=current_node,
        walked_path=["model", "bs"],
        coupled_sweep_mapping=get_coupled_sweep_mapping,
    )
    assert config.get("model").get("bs") == 32
    # other fields remain unchanged
    assert config.get("trainer").get("foo_ref") == get_suggestions_with_best_trial.get(
        ("trainer")
    ).get("foo_ref")


def test_resolve_references_all(
    get_suggestions_with_best_trial, get_coupled_sweep_mapping
):
    config = copy.deepcopy(get_suggestions_with_best_trial)
    tune.resolve_references(
        config=config,
        node=config,
        walked_path=[],
        coupled_sweep_mapping=get_coupled_sweep_mapping,
    )
    # check coupled-sweep resolved
    assert config.get("model").get("bs") == 32
    assert config.get("model").get("bar")[0].get("x") == -1
    assert config.get("model").get("baz")[0].get("x") == -10
    # check reference resolved
    assert config.get("trainer").get("foo_ref") == 1


def test_load_yaml_invalid(tmp_path):
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(None)
    with pytest.raises(FileNotFoundError):
        tune.load_yaml("")
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(1)
    with pytest.raises(FileNotFoundError):
        tune.load_yaml(tmp_path / "non_existent.yaml")


def test_load_yaml_valid(yaml_text, tmp_path):
    # create a temporary YAML file
    yaml_file = tmp_path / "sample.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_text)

    config = tune.load_yaml(yaml_file)

    assert "model" in config
    assert "trainer" in config
    assert config["model"]["name"] == "test_model"
    assert config["trainer"]["lr"]["type"] == "range"


# def test_build_search_space_with_dependencies_tune_all(
#     get_config_file, get_dependencies_file, get_fixed_trial
# ):
#     search_space = tune.build_search_space_with_dependencies(
#         get_config_file,
#         get_dependencies_file,
#         get_fixed_trial,
#         tune_model=True,
#         tune_training=True,
#         base_settings_file=None,
#     )

#     expected_search_space = {
#         "model": {
#             "gcn_net": [
#                 {"in_dim": 16, "out_dim": 32, "norm_args": [32]},
#                 {"in_dim": 32, "out_dim": 64, "norm_args": [64]},
#             ],
#             "classifier": {"in_dim": 64},
#             "num_layers": 2,
#             "activation": "tanh",
#             "name": "MyModel",
#         },
#         "training": {
#             "lr": 0.0001,
#             "batch_size": 32,
#         },
#     }
#     assert search_space == expected_search_space


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


def test_get_best_trial(tmp_path):
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
    tune.get_best_trial(study, output_file)
    loaded_config = tune.load_yaml(output_file)
    assert loaded_config == {"lr": 0.01, "n_layers": 3}


# def test_save_best_config(tmp_path):
#     built_search_space = {
#         "model": {
#             "nn": [
#                 {
#                     "in_dim": "waiting",
#                     "out_dim": "waiting",
#                     "dropout": "waiting",
#                 },
#                 {
#                     "in_dim": "waiting",
#                     "out_dim": "waiting",
#                     "dropout": "waiting",
#                 },
#             ]
#         }
#     }
#     best_trial = {
#         "model.nn.0.in_dim": 784,
#         "model.nn.0.out_dim": 32,
#         "model.nn.0.dropout": 0.5,
#         "model.nn.1.in_dim": 32,
#         "model.nn.1.out_dim": 32,
#         "model.nn.1.dropout": 0.1,
#     }
#     depmap = {
#         "model": {
#             "nn": [
#                 {},
#                 {
#                     "in_dim": "model.nn[0].out_dim",
#                 },
#             ]
#         }
#     }
#     built_search_space_file = tmp_path / "built_search_space.yaml"
#     with open(built_search_space_file, "w") as f:
#         yaml.safe_dump(built_search_space, f)
#     best_trial_file = tmp_path / "best_trial.yaml"
#     with open(best_trial_file, "w") as f:
#         yaml.safe_dump(best_trial, f)
#     depmap_file = tmp_path / "depmap.yaml"
#     with open(depmap_file, "w") as f:
#         yaml.safe_dump(depmap, f)
#     output_file = tmp_path / "best_config.yaml"

#     tune.save_best_config(
#         built_search_space_file,
#         best_trial_file,
#         depmap_file,
#         output_file,
#     )

#     with open(output_file, "r") as f:
#         best_config = yaml.safe_load(f)

#     expected_config = {
#         "model": {
#             "nn": [
#                 {
#                     "in_dim": 784,
#                     "out_dim": 32,
#                     "dropout": 0.5,
#                 },
#                 {
#                     "in_dim": 32,
#                     "out_dim": 32,
#                     "dropout": 0.1,
#                 },
#             ]
#         }
#     }
#     assert best_config == expected_config
