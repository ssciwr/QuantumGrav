import QuantumGrav as QG
import pytest
import yaml
import torch_geometric
import numpy as np


@pytest.fixture(scope="session")
def yaml_text_nonsweep():
    yaml_text = """
        model:
            name: test_model
            layers: 1
            type: !pyobject QuantumGrav.GNNBlock
            convtype: !pyobject torch_geometric.nn.SAGEConv
            bs: 16
            lr: 0.1
            foo:
                - 3

                -
                    x: 1
                    y: 2
            bar:
                - x: 1
            baz:
                - x: 2

        trainer:
            epochs: 2
        """
    return yaml_text


def test_range_inclusive():
    values = QG.config_utils.range_inclusive(1, 5, 1)
    assert np.array_equal(values, np.array([1, 2, 3, 4, 5]))

    values = QG.config_utils.range_inclusive(0, 1, 0.2)
    assert np.allclose(values, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

    values = QG.config_utils.range_inclusive(5, 2, -1)
    assert np.array_equal(values, np.array([5, 4, 3, 2]))

    values = QG.config_utils.range_inclusive(0.5, 0.1, -0.2)
    assert np.allclose(values, np.array([0.5, 0.3, 0.1]))

    values = QG.config_utils.range_inclusive(1, 6, 2)
    assert np.array_equal(values, np.array([1, 3, 5]))


def test_read_yaml(yaml_text):
    # Parse a YAML string using the custom tags and ensure structures are constructed correctly
    loader = QG.config_utils.get_loader()
    cfg = yaml.load(yaml_text, Loader=loader)

    # sweep nodes
    assert isinstance(cfg["model"]["layers"], dict)
    assert cfg["model"]["layers"]["type"] == "sweep"
    assert cfg["model"]["layers"]["values"] == [1, 2]
    assert cfg["model"]["foo"][0]["x"] == 3
    assert cfg["model"]["foo"][0]["y"] == 5
    assert cfg["model"]["foo"][1]["x"]["type"] == "sweep"
    assert cfg["model"]["foo"][1]["x"]["values"] == [1, 2]
    assert cfg["model"]["foo"][1]["y"] == 2

    # coupled-sweep nodes
    cs = cfg["model"]["bs"]
    assert isinstance(cs, dict)
    assert cs["type"] == "coupled-sweep"
    # target path is split into components
    assert cs["target"] == ["model", "layers"]
    assert cs["values"] == [16, 32]
    assert cfg["model"]["bar"][0]["x"]["type"] == "coupled-sweep"
    assert cfg["model"]["bar"][0]["x"]["target"] == ["model", "foo", 1, "x"]
    assert cfg["model"]["bar"][0]["x"]["values"] == [-1, -2]
    assert cfg["model"]["baz"][0]["x"]["type"] == "coupled-sweep"
    assert cfg["model"]["baz"][0]["x"]["target"] == ["model", "foo", 1, "x"]
    assert cfg["model"]["baz"][0]["x"]["values"] == [-10, -20]

    # range nodes
    rn = cfg["trainer"]["epochs"]
    assert isinstance(rn, dict)
    assert rn["type"] == "range"
    assert list(rn["values"]) == [1, 3, 5]
    assert rn["tune_values"] == (1, 6, 2)

    rn_lr = cfg["trainer"]["lr"]
    assert isinstance(rn_lr, dict)
    assert rn_lr["type"] == "range"
    assert len(rn_lr["values"]) == 4
    assert all(isinstance(v, float) for v in rn_lr["values"])
    assert rn_lr["values"][0] >= 1e-5 and rn_lr["values"][-1] <= 1e-2
    assert rn_lr["tune_values"] == (1e-5, 1e-2, True)

    rn_drop_rate = cfg["trainer"]["drop_rate"]
    assert isinstance(rn_drop_rate, dict)
    assert rn_drop_rate["type"] == "range"
    assert np.allclose(rn_drop_rate["values"], [0.1, 0.3, 0.5])
    assert np.allclose(rn_drop_rate["tune_values"], (0.1, 0.5, 0.2))

    # reference node
    ref = cfg["trainer"]["foo_ref"]
    assert isinstance(ref, dict)
    assert ref["type"] == "reference"
    assert ref["target"] == ["model", "foo", 1, "x"]

    # type nodes
    tn = cfg["model"]["type"]
    assert tn == QG.GNNBlock

    tn = cfg["model"]["convtype"]
    assert tn == torch_geometric.nn.SAGEConv

    yaml_text = """
        model:
            layers: !sweep
                values: []
    """
    cfg = yaml.load(yaml_text, Loader=loader)


def test_read_yaml_throws():
    broken_yaml_text = """
        model:
            type: !pyobject QuantumGrav.DoesNotExist
        """
    loader = QG.config_utils.get_loader()
    with pytest.raises(
        ValueError, match="Could not load name DoesNotExist from QuantumGrav"
    ):
        yaml.load(broken_yaml_text, Loader=loader)

    broken_yaml_text = """
        model:
            type: !pyobject DoesNotExist.Irrelevant
        """
    with pytest.raises(ValueError, match="Importing module DoesNotExist unsuccessful"):
        yaml.load(broken_yaml_text, Loader=loader)


def test_convert_to_pyobject_tags():
    config = {
        "model": {
            "type": QG.gnn_block.GNNBlock,
            "convtype": torch_geometric.nn.conv.sage_conv.SAGEConv,
            "name": "test_model",
            "layers": 2,
            "bs": 32,
        },
        "trainer": {"epochs": 5},
    }
    config_with_tags = QG.config_utils.convert_to_pyobject_tags(config)

    # check if the types are converted to pyobject tags
    assert (
        config_with_tags["model"]["type"] == "!pyobject QuantumGrav.gnn_block.GNNBlock"
    )
    assert (
        config_with_tags["model"]["convtype"]
        == "!pyobject torch_geometric.nn.conv.sage_conv.SAGEConv"
    )

    # check if other values remain unchanged
    assert config_with_tags["model"]["name"] == "test_model"
    assert config_with_tags["model"]["layers"] == 2
    assert config_with_tags["model"]["bs"] == 32
    assert config_with_tags["trainer"]["epochs"] == 5


def test_initialize_config_handler_nonsweep(yaml_text_nonsweep):
    loader = QG.config_utils.get_loader()
    cfg = yaml.load(yaml_text_nonsweep, Loader=loader)
    ch = QG.ConfigHandler(cfg)
    run_cfgs = ch.run_configs
    assert len(run_cfgs) == 1


def test_initialize_config_handler(yaml_text):
    loader = QG.config_utils.get_loader()
    base_cfg = yaml.load(yaml_text, Loader=loader)

    ch = QG.ConfigHandler(base_cfg)
    run_cfgs = ch.run_configs

    assert isinstance(run_cfgs, list)
    assert len(run_cfgs) == 12

    # Verify coupling: for layers=1 -> bs=16, layers=2 -> bs=32
    observed = set()
    for i, rc in enumerate(run_cfgs):
        layers = rc["model"]["layers"]
        bs = rc["model"]["bs"]
        lr = rc["model"]["lr"]
        foo = rc["model"]["foo"][1]
        bar = rc["model"]["bar"][0]
        baz = rc["model"]["baz"][0]

        assert layers in [1, 2]
        assert lr in [0.1, 0.01, 0.001]
        assert bs == (16 if layers == 1 else 32)
        assert foo["x"] in [1, 2]
        assert bar["x"] in [-1, -2]
        assert baz["x"] in [-10, -20]

        observed.add((layers, bs, lr, foo["x"], bar["x"], baz["x"]))

        name = rc["model"]["name"]
        assert name.startswith("test_model_run_")
        assert name.endswith(f"run_{i}")
    # Ensure all combinations of (layers, lr) appear with the correct coupling for bs
    expected = set()
    for layers, bs in [(1, 16), (2, 32)]:
        for lr in [0.1, 0.01, 0.001]:
            for foo_x, bar_x, baz_x in [(1, -1, -10), (2, -2, -20)]:
                expected.add((layers, bs, lr, foo_x, bar_x, baz_x))
    assert observed == expected


def test_initialize_config_handler_fails():
    # Coupled-sweep must match the length of its target sweep; otherwise an IndexError should be raised
    yaml_text = """
        model:
            layers: !sweep
                values: [1, 2]
            bs: !coupled-sweep
                target: model.layers
                values: [16, 32, 64]  # length mismatch
        """

    loader = QG.config_utils.get_loader()
    bad_cfg = yaml.load(yaml_text, Loader=loader)

    with pytest.raises(ValueError, match="Incompatible lengths for coupled-sweep"):
        QG.ConfigHandler(bad_cfg)
