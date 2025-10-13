import QuantumGrav as QG
import pytest
import yaml
import torch_geometric


@pytest.fixture(scope="session")
def yaml_text():
    yaml_text = """
        model:
            name: test_model
            layers: !sweep
                values: [1, 2]

            type: !pyobject QuantumGrav.GNNBlock
            convtype: !pyobject torch_geometric.nn.SAGEConv
            bs: !coupled-sweep
                target: model.layers
                values: [16, 32]
            lr: !sweep
                values: [0.1, 0.01, 0.001]
            foo: 
                - 
                    x: 3 
                    y: 5 
                - 
                    x: !sweep 
                        values: [1, 2]
                    y: 2
            bar: 
                - x: !coupled-sweep 
                    target: model.foo[1].x 
                    values: [-1, -2]
            baz: 
                - x: !coupled-sweep 
                    target: model.foo[1].x
                    values: [-10, -20]
            
        trainer:
            epochs: !range
                start: 1
                stop: 6
                step: 2
        """
    return yaml_text


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
    assert cfg["model"]["baz"][0]["x"]["values"] == [-2, -4]

    # range nodes
    rn = cfg["trainer"]["epochs"]
    assert isinstance(rn, dict)
    assert rn["type"] == "range"
    assert list(rn["values"]) == [1, 3, 5]

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
        assert baz["x"] in [-2, -4]

        observed.add((layers, bs, lr, foo["x"], bar["x"]))

        name = rc["model"]["name"]
        assert name.startswith("test_model/run_")
        assert name.endswith(f"run_{i}")
    # Ensure all combinations of (layers, lr) appear with the correct coupling for bs
    expected = set()
    for layers, bs in [(1, 16), (2, 32)]:
        for lr in [0.1, 0.01, 0.001]:
            for foo_x, bar_x, baz_x in [(1, -1, -2), (2, -2, -4)]:
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

    with pytest.raises(IndexError):
        QG.ConfigHandler(bad_cfg)
