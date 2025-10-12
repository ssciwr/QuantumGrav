import QuantumGrav as QG
from QuantumGrav import ConfigHandler
from QuantumGrav.config_utils import get_loader
import pytest
import yaml


def test_read_yaml():
        # Parse a YAML string using the custom tags and ensure structures are constructed correctly
        yaml_text = """
        model:
            layers: !sweep
                values: [1, 2]
            bs: !coupled-sweep
                target: model.layers
                values: [16, 32]
            lr: !sweep
                values: [0.1, 0.01, 0.001]
        trainer:
            epochs: !range
                start: 1
                stop: 4
                step: 1
        """

        loader = get_loader()
        cfg = yaml.load(yaml_text, Loader=loader)

        # sweep nodes
        assert isinstance(cfg["model"]["layers"], dict)
        assert cfg["model"]["layers"]["type"] == "sweep"
        assert cfg["model"]["layers"]["values"] == [1, 2]

        # coupled-sweep nodes
        cs = cfg["model"]["bs"]
        assert isinstance(cs, dict)
        assert cs["type"] == "coupled-sweep"
        # target path is split into components
        assert cs["target"] == ["model", "layers"]
        assert cs["values"] == [16, 32]

        # range nodes
        rn = cfg["trainer"]["epochs"]
        assert isinstance(rn, dict)
        assert rn["type"] == "range"
        assert list(rn["values"]) == [1, 2, 3]


def test_initialize_config_handler():
        # Build a config with two sweeps and one coupled-sweep to verify run_configs are expanded correctly
        yaml_text = """
        model:
            layers: !sweep
                values: [1, 2]
            bs: !coupled-sweep
                target: model.layers
                values: [16, 32]
            lr: !sweep
                values: [0.1, 0.01, 0.001]
        trainer:
            epochs: !range
                start: 1
                stop: 4
                step: 1
        """

        loader = get_loader()
        base_cfg = yaml.load(yaml_text, Loader=loader)

        ch = ConfigHandler(base_cfg)
        run_cfgs = ch.run_configs

        # Expect cartesian product of layers (2) x lr (3) = 6 configurations
        assert isinstance(run_cfgs, list)
        assert len(run_cfgs) == 6

        # Verify coupling: for layers=1 -> bs=16, layers=2 -> bs=32
        observed = set()
        for rc in run_cfgs:
                layers = rc["model"]["layers"]
                bs = rc["model"]["bs"]
                lr = rc["model"]["lr"]

                assert layers in [1, 2]
                assert lr in [0.1, 0.01, 0.001]
                assert bs == (16 if layers == 1 else 32)

                observed.add((layers, bs, lr))

        # Ensure all combinations of (layers, lr) appear with the correct coupling for bs
        expected = set()
        for layers, bs in [(1, 16), (2, 32)]:
                for lr in [0.1, 0.01, 0.001]:
                        expected.add((layers, bs, lr))
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

        loader = get_loader()
        bad_cfg = yaml.load(yaml_text, Loader=loader)

        with pytest.raises(IndexError):
            ConfigHandler(bad_cfg)