import pytest
import torch
import torch_geometric
from torch_geometric.data import Data
from QuantumGrav.train import Snapshot

import QuantumGrav as QG
import numpy as np
import jsonschema
from pathlib import Path
import re
import pandas as pd
from copy import deepcopy
import logging

torch.multiprocessing.set_start_method("spawn", force=True)  # for dataloader


# data transform functions
def cat1(x):
    return torch.cat(x, dim=1)


# evaluator helpers (for DefaultValidator/DefaultTester)
def eval_loss(x: dict[int, torch.Tensor], data: Data) -> torch.Tensor:
    """Loss used by DefaultEvaluator: outputs + data -> loss.

    Aggregates MSE across all active task outputs.
    """
    all_loss = torch.zeros(1)
    for _, task_output in x.items():
        loss = torch.nn.MSELoss()(task_output, data.y.to(torch.float32))  # type: ignore
        all_loss += loss
    return all_loss


def monitor_dummy(preds, targets):
    """Simple monitor that returns zero as a scalar."""
    return 0.0


# this is needed for testing the full training loop
class DummyEvaluator(QG.Evaluator):
    def __init__(self):
        self.data = pd.DataFrame(columns=["loss", "other_loss"])

    def validate(self, model, data_loader):
        # Dummy validate logic
        losses = torch.rand(10)
        avg1 = losses.mean().item()
        avg2 = losses.mean().item()
        self.data.loc[len(self.data), "loss"] = avg1
        self.data.loc[len(self.data) - 1, "other_loss"] = avg2

    def test(self, model, data_loader):
        # Dummy test logic
        losses = torch.rand(10)
        avg1 = losses.mean().item()
        avg2 = losses.mean().item()
        self.data.loc[len(self.data), "loss"] = avg1
        self.data.loc[len(self.data) - 1, "other_loss"] = avg2

    def report(self, losses: list):  # type: ignore
        print("DummyEvaluator report:", losses, self.data.tail(1))


def compute_loss(
    x: dict[int, torch.Tensor], data: Data, trainer: QG.Trainer
) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    all_loss = torch.zeros(1)
    for _, task_output in x.items():
        loss = torch.nn.MSELoss()(task_output, data.y.to(torch.float32))  # type: ignore
        all_loss += loss
    return all_loss


# test fixtures
@pytest.fixture
def tmppath(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoints")
    return path


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
                    [(64, 24), (24, 18), (18, 1)],
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
                    [(64, 24), (24, 18), (18, 1)],
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
        "aggregate_pooling_type": cat1,
        "active_tasks": {0: True, 1: True},
    }
    return config


@pytest.fixture
def config(model_config_eval, tmppath, create_data_zarr, read_data):
    datadir, datafiles = create_data_zarr
    cfg = {
        "training": {
            "seed": 42,
            # training loop
            "device": "cpu",
            "checkpoint_at": 20,
            "path": str(tmppath),
            # optimizer
            "optimizer_type": torch.optim.Adam,
            "optimizer_args": [],
            "optimizer_kwargs": {"lr": 0.001, "weight_decay": 0.0},
            # training loader
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": True,
            "num_epochs": 13,
            # "prefetch_factor": 2,
        },
        "data": {
            "pre_transform": lambda x: x,
            "transform": lambda x: x,
            "pre_filter": lambda x: True,
            "reader": read_data,
            "files": [str(f) for f in datafiles],
            "output": str(datadir),
            "validate_data": True,
            "n_processes": 2,
            "chunksize": 10,
            "shuffle": False,
            "split": [0.8, 0.1, 0.1],
        },
        "model": model_config_eval,
        "criterion": compute_loss,
        "validation": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": True,
            "validator": {
                "type": DummyEvaluator,
                "args": [],
                "kwargs": {},
            },
        },
        "testing": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": False,
            "tester": {
                "type": DummyEvaluator,
                "args": [],
                "kwargs": {},
            },
        },
        "early_stopping": {
            "type": QG.early_stopping.DefaultEarlyStopping,
            "args": [
                {
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
                12,
            ],
            "kwargs": {
                "mode": "any",
            },
        },
    }

    return cfg


@pytest.fixture
def config_with_default_evaluators(config):
    cfg = deepcopy(config)
    cfg["validation"]["validator"] = {
        "type": QG.Validator,
        "args": [
            "cpu",
            eval_loss,
            [
                {
                    "name": "loss",
                    "monitor": monitor_dummy,
                },
                {"name": "other_loss", "monitor": monitor_dummy},
            ],
        ],
        "kwargs": {},
    }
    cfg["testing"]["tester"] = {
        "type": QG.Tester,
        "args": [
            "cpu",
            eval_loss,
            [
                {
                    "name": "loss",
                    "monitor": monitor_dummy,
                },
                {"name": "other_loss", "monitor": monitor_dummy},
            ],
        ],
        "kwargs": {},
    }

    cfg["early_stopping"] = {
        "type": QG.early_stopping.DefaultEarlyStopping,
        "args": [
            {
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
            12,
        ],
        "kwargs": {
            "mode": "any",
        },
    }

    return cfg


@pytest.fixture
def config_with_data(config, create_data_zarr, read_data):
    datadir, datafiles = create_data_zarr
    cfg = deepcopy(config)
    cfg["data"] = {
        "pre_transform": lambda x: x,
        "transform": lambda x: x,
        "pre_filter": lambda x: True,
        "reader": read_data,
        "files": [str(f) for f in datafiles],
        "output": str(datadir),
        "validate_data": True,
        "n_processes": 2,
        "chunksize": 10,
        "shuffle": False,
    }

    return cfg


@pytest.fixture
def config_with_scheduler(config):
    cfg = deepcopy(config)
    cfg["training"]["lr_scheduler_type"] = torch.optim.lr_scheduler.LinearLR
    cfg["training"]["lr_scheduler_args"] = [0.2, 0.8]
    cfg["training"]["lr_scheduler_kwargs"] = {"total_iters": 10, "last_epoch": -1}
    return cfg


@pytest.fixture
def broken_config(model_config_eval):
    return {
        "training": {
            "seed": 42,
            # training loop
            "device": "cpu",
            "early_stopping_patience": 10,
            "checkpoint_at": 10,
            # optimizer
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            # training loader
            "batch_size": 4,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
            "prefetch_factor": 2,
        },
        "model": model_config_eval,
        "criterion": compute_loss,
        # validation is missing -> broken
        "testing": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "prefetch_factor": 1,
            "shuffle": False,
        },
    }


def make_injected_trainer(
    config,
    data_path: Path,
    *,
    validator=None,
    tester=None,
    early_stopper=None,
):
    logger = logging.getLogger(f"{__name__}.{data_path.name}")
    logger.setLevel(logging.CRITICAL)

    return QG.Trainer(
        config=config,
        logger=logger,
        criterion=config["criterion"],
        model=None,
        optimizer=None,
        seed=config["training"]["seed"],
        device=torch.device(config["training"]["device"]),
        data_path=data_path,
        lr_scheduler=None,
        early_stopper=early_stopper,
        validator=validator,
        tester=tester,
        apply_model=config.get("apply_model"),
    )


def make_loader_factory(config):
    """Create a data loader factory for tests."""
    return QG.DataLoaderFactory.from_config(config)


def make_data_config(base_config, files, output):
    """Create a dataset config for a specific set of files."""
    data_config = deepcopy(base_config)
    data_config["files"] = [str(file) for file in files]
    data_config["output"] = str(output)
    return data_config


def test_trainer_creation_works(config):
    trainer = QG.Trainer.from_config(config)

    assert trainer.config == config
    assert trainer.criterion is compute_loss
    assert trainer.apply_model is None
    assert isinstance(trainer.early_stopper, QG.DefaultEarlyStopping)
    assert isinstance(trainer.validator, DummyEvaluator)
    assert isinstance(trainer.tester, DummyEvaluator)

    assert trainer.device == torch.device("cpu")
    assert trainer.seed == config["training"]["seed"]
    assert trainer.epoch == 0
    assert trainer.checkpoint_at == config["training"].get("checkpoint_at", None)
    assert trainer.data_path.exists()
    assert trainer.checkpoint_path == trainer.data_path / "checkpoints"

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer.model, QG.GNNModel)
    assert trainer.lr_scheduler is None


def test_trainer_model_instantiation_works(config):
    trainer = QG.Trainer.from_config(config)
    original_model = trainer.model
    trainer.initialize_model()

    assert isinstance(trainer.model, QG.GNNModel)
    assert trainer.model is not original_model
    assert isinstance(trainer.validator, DummyEvaluator)
    assert isinstance(trainer.tester, DummyEvaluator)


def test_trainer_creation_with_default_evaluators(config_with_default_evaluators):
    trainer = QG.Trainer.from_config(config_with_default_evaluators)

    assert isinstance(trainer.validator, QG.Validator)
    assert isinstance(trainer.tester, QG.Tester)
    assert trainer.device == torch.device("cpu")
    assert trainer.seed == config_with_default_evaluators["training"]["seed"]


def test_trainer_init_allows_optional_components(config, tmppath):
    trainer = make_injected_trainer(
        config,
        tmppath / "manual_trainer",
        validator=None,
        tester=None,
        early_stopper=None,
    )

    assert trainer.validator is None
    assert trainer.tester is None
    assert trainer.early_stopper is None
    assert isinstance(trainer.model, QG.GNNModel)
    assert isinstance(trainer.optimizer, torch.optim.Adam)


def test_trainer_optimizer_instantiation_works(config):
    trainer = QG.Trainer.from_config(config)
    original_optimizer = trainer.optimizer
    trainer.initialize_optimizer()

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer is not original_optimizer


def test_trainer_creation_broken(broken_config):
    with pytest.raises(
        jsonschema.ValidationError,
        match="'validation' is a required property",
    ):
        QG.Trainer.from_config(broken_config)


def test_dataloader_factory_schema_owns_data_config():
    assert "data" not in QG.Trainer.schema["properties"]
    assert "data" in QG.DataLoaderFactory.schema["properties"]
    assert "data" in QG.DataLoaderFactory.schema["properties"]["training"]["properties"]
    assert (
        "data" in QG.DataLoaderFactory.schema["properties"]["validation"]["properties"]
    )
    assert "data" in QG.DataLoaderFactory.schema["properties"]["testing"]["properties"]


def test_trainer_init_model(config):
    trainer = QG.Trainer.from_config(config)
    model = trainer.initialize_model()
    assert model is not None
    assert isinstance(model, QG.GNNModel)


def test_trainer_init_optimizer(config):
    trainer = QG.Trainer.from_config(config)
    trainer.optimizer = None
    trainer.initialize_optimizer()
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_trainer_init_optimizer_fails(config):
    trainer = QG.Trainer.from_config(config)
    trainer.model = None
    with pytest.raises(
        RuntimeError, match="Model must be initialized before initializing optimizer."
    ):
        trainer.initialize_optimizer()


def test_trainer_init_lr_scheduler(config_with_scheduler):
    trainer = QG.Trainer.from_config(config_with_scheduler)

    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.LinearLR)
    assert trainer.lr_scheduler.start_factor == 0.2
    assert trainer.lr_scheduler.end_factor == 0.8
    assert trainer.lr_scheduler.total_iters == 10
    assert trainer.lr_scheduler.last_epoch == 0


def test_trainer_init_lr_scheduler_fails(config_with_scheduler):
    trainer = QG.Trainer.from_config(config_with_scheduler)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    with pytest.raises(
        RuntimeError,
        match="Optimizer must be initialized before initializing learning rate scheduler.",
    ):
        trainer.initialize_lr_scheduler()


def test_trainer_prepare_dataloader(make_dataset, config):
    factory = make_loader_factory(config)

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )

    assert len(train_loader) == 3
    assert len(val_loader) == 1
    assert len(test_loader) == 2

    for batch in train_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (60, 2)

    for batch in val_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    for batch in test_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)


def test_trainer_prepare_dataloader_broken(make_dataset, config):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Split ratios must sum to 1.0. Provided split: [0.9, 0.2, 0.1]"
        ),
    ):
        config["data"]["split"] = [0.9, 0.2, 0.1]
        factory = make_loader_factory(config)
        factory.prepare_dataloaders(make_dataset)

    with pytest.raises(ValueError, match=re.escape("validation size cannot be 0")):
        config["data"]["split"] = [0.95, 0.01, 0.04]
        factory = make_loader_factory(config)
        factory.prepare_dataloaders(make_dataset)

    with pytest.raises(ValueError, match=re.escape("test size cannot be 0")):
        config["data"]["split"] = [0.85, 0.14, 0.01]
        factory = make_loader_factory(config)
        factory.prepare_dataloaders(
            make_dataset,
        )


def test_trainer_prepare_dataloader_with_dataconf(config_with_data):
    factory = make_loader_factory(config_with_data)

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        split=[0.8, 0.1, 0.1]
    )

    for batch in train_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (60, 2)

    for batch in val_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    for batch in test_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    config_with_data["data"]["shuffle"] = True
    factory = make_loader_factory(config_with_data)

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        split=[0.8, 0.1, 0.1]
    )

    for batch in train_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (60, 2)

    for batch in val_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    for batch in test_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    config_with_data["data"]["subset"] = 0.5
    factory = make_loader_factory(config_with_data)

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        split=[0.6, 0.2, 0.2]
    )

    for batch in train_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (60, 2)

    for batch in val_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)

    for batch in test_loader:
        assert isinstance(batch, Data)
        assert batch.x is not None
        assert batch.x.shape == (15, 2)


def test_trainer_prepare_dataloader_with_shared_train_validation_data(
    config, create_data_zarr
):
    datadir, datafiles = create_data_zarr
    cfg = deepcopy(config)
    cfg["data"] = make_data_config(
        cfg["data"], datafiles[:2], Path(datadir) / "shared_train_validation"
    )
    cfg["data"]["split"] = [0.8, 0.2]
    cfg["testing"]["data"] = make_data_config(
        cfg["data"], [datafiles[2]], Path(datadir) / "shared_testing"
    )

    train_loader, val_loader, test_loader = make_loader_factory(
        cfg
    ).prepare_dataloaders()

    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 5
    assert set(train_loader.dataset.indices).isdisjoint(val_loader.dataset.indices)


def test_trainer_prepare_dataloader_with_stage_local_data(config, create_data_zarr):
    datadir, datafiles = create_data_zarr
    cfg = deepcopy(config)
    cfg.pop("data")
    cfg["training"]["data"] = make_data_config(
        config["data"], [datafiles[0]], Path(datadir) / "training_dataset"
    )
    cfg["training"]["data"].pop("split", None)
    cfg["validation"]["data"] = make_data_config(
        config["data"], [datafiles[1]], Path(datadir) / "validation_dataset"
    )
    cfg["validation"]["data"].pop("split", None)
    cfg["testing"]["data"] = make_data_config(
        config["data"], [datafiles[2]], Path(datadir) / "testing_dataset"
    )
    cfg["testing"]["data"].pop("split", None)

    train_loader, val_loader, test_loader = make_loader_factory(
        cfg
    ).prepare_dataloaders()

    assert len(train_loader.dataset) == 5
    assert len(val_loader.dataset) == 5
    assert len(test_loader.dataset) == 5


def test_trainer_prepare_dataloader_rejects_mixed_data_configs(
    config, create_data_zarr
):
    datadir, datafiles = create_data_zarr

    cfg = deepcopy(config)
    cfg.pop("data")
    cfg["training"]["data"] = make_data_config(
        config["data"], [datafiles[0]], Path(datadir) / "broken_training_dataset"
    )
    with pytest.raises(ValueError, match="Unsupported data config"):
        make_loader_factory(cfg).prepare_dataloaders()

    cfg = deepcopy(config)
    cfg.pop("data")
    cfg["validation"]["data"] = make_data_config(
        config["data"], [datafiles[1]], Path(datadir) / "broken_validation_dataset"
    )
    with pytest.raises(ValueError, match="Unsupported data config"):
        make_loader_factory(cfg).prepare_dataloaders()

    cfg = deepcopy(config)
    cfg.pop("data")
    cfg["training"]["data"] = make_data_config(
        config["data"], [datafiles[0]], Path(datadir) / "broken_training_stage_dataset"
    )
    cfg["testing"]["data"] = make_data_config(
        config["data"], [datafiles[2]], Path(datadir) / "broken_testing_stage_dataset"
    )
    with pytest.raises(ValueError, match="Unsupported data config"):
        make_loader_factory(cfg).prepare_dataloaders()

    cfg = deepcopy(config)
    cfg["validation"]["data"] = make_data_config(
        config["data"],
        [datafiles[1]],
        Path(datadir) / "broken_mixed_validation_dataset",
    )
    with pytest.raises(ValueError, match="Unsupported data config"):
        make_loader_factory(cfg).prepare_dataloaders()


def test_trainer_train_epoch(make_dataset, config):
    trainer = QG.Trainer.from_config(config)
    assert trainer.model is not None
    assert trainer.optimizer is not None

    train_loader, _, _ = make_loader_factory(config).prepare_dataloaders(make_dataset)
    trainer.model.train()

    eval_data = trainer._run_train_epoch(trainer.model, trainer.optimizer, train_loader)

    assert trainer.model.training is True
    assert len(eval_data) == len(train_loader)


def test_trainer_check_model_status(config):
    trainer = QG.Trainer.from_config(config)

    loss = np.random.rand(10).tolist()
    other__loss = np.random.rand(10).tolist()
    data = pd.DataFrame({"loss": loss, "other_loss": other__loss})

    trainer.epoch = 1
    trainer.early_stopper = None
    saved = trainer._check_model_status(data)
    assert saved is False

    # returns true when early stopping is triggered
    trainer.early_stopper = lambda x: True

    saved = trainer._check_model_status(data)

    assert saved is True

    file_content = [f.name for f in trainer.data_path.iterdir()]
    assert "config.yaml" in file_content
    assert "checkpoints" in file_content


def test_trainer_save_checkpoint_writes_snapshot(config):
    trainer = QG.Trainer.from_config(config)

    trainer.save_checkpoint()

    assert (trainer.checkpoint_path / "epoch_0").exists()


def test_snapshot_from_trainer_collects_expected_state(config):
    trainer = QG.Trainer.from_config(config)

    snapshot = Snapshot.from_trainer(trainer)

    assert snapshot.epoch == trainer.epoch
    assert snapshot.path == trainer.checkpoint_path / f"epoch_{trainer.epoch}"
    assert snapshot.config_path == trainer.data_path / "config.yaml"
    assert snapshot.model_state_dict is not None
    assert snapshot.optimizer_state_dict is not None
    assert snapshot.lr_scheduler_state_dict is None
    assert snapshot.validator_state_dict is not None
    assert snapshot.tester_state_dict is not None
    assert snapshot.early_stopping_state_dict is not None


def test_snapshot_save_and_load_round_trip(
    config_with_default_evaluators, make_dataset
):
    config = deepcopy(config_with_default_evaluators)
    config.pop("data", None)
    config["training"]["num_epochs"] = 4
    trainer = QG.Trainer.from_config(config)

    assert isinstance(trainer.validator, QG.Validator)
    assert isinstance(trainer.tester, QG.Tester)
    assert isinstance(trainer.early_stopper, QG.DefaultEarlyStopping)

    trainer.epoch = 3
    trainer.validator.data.loc[0] = {
        "loss_avg": 0.2,
        "loss_min": 0.1,
        "loss_max": 0.3,
        "loss": 0.22,
        "other_loss": 0.55,
    }
    trainer.tester.data.loc[0] = {
        "loss_avg": 0.5,
        "loss_min": 0.4,
        "loss_max": 0.8,
        "loss": 0.52,
        "other_loss": 0.61,
    }
    trainer.early_stopper.current_patience = 4
    trainer.early_stopper.tasks[0]["best_score"] = 0.123
    trainer.early_stopper.tasks[0]["current_grace_period"] = 3
    trainer.early_stopper.tasks[0]["found_better"] = True
    trainer.early_stopper.tasks[1]["best_score"] = 9.876
    trainer.early_stopper.tasks[1]["current_grace_period"] = 1
    trainer.early_stopper.tasks[1]["found_better"] = False

    original = Snapshot.from_trainer(trainer)
    original.path = trainer.checkpoint_path / "epoch_3_roundtrip"
    original.save()

    loaded_trainer = QG.Trainer.load_checkpoint(
        trainer.checkpoint_path / "epoch_3_roundtrip"
    )

    assert isinstance(loaded_trainer.validator, QG.Validator)
    assert isinstance(loaded_trainer.tester, QG.Tester)
    assert isinstance(loaded_trainer.early_stopper, QG.DefaultEarlyStopping)
    assert loaded_trainer.epoch == trainer.epoch

    assert loaded_trainer.model is not None
    assert loaded_trainer.model.state_dict().keys() == trainer.model.state_dict().keys()
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(loaded_trainer.model.state_dict()[key], value)

    assert loaded_trainer.optimizer is not None
    assert loaded_trainer.optimizer.state_dict() == trainer.optimizer.state_dict()

    assert loaded_trainer.validator is not None
    pd.testing.assert_frame_equal(
        loaded_trainer.validator.data.reset_index(drop=True),
        trainer.validator.data.reset_index(drop=True),
    )

    assert loaded_trainer.tester is not None
    pd.testing.assert_frame_equal(
        loaded_trainer.tester.data.reset_index(drop=True),
        trainer.tester.data.reset_index(drop=True),
    )

    assert loaded_trainer.early_stopper is not None
    assert (
        loaded_trainer.early_stopper.current_patience
        == trainer.early_stopper.current_patience
    )
    for task_id, task in trainer.early_stopper.tasks.items():
        loaded_task = loaded_trainer.early_stopper.tasks[task_id]
        assert loaded_task["best_score"] == task["best_score"]
        assert loaded_task["current_grace_period"] == task["current_grace_period"]
        assert loaded_task["found_better"] == task["found_better"]

    train_loader, validation_loader, _ = make_loader_factory(
        config_with_default_evaluators
    ).prepare_dataloaders(make_dataset, split=[0.8, 0.1, 0.1])
    training_data, validation_data = loaded_trainer.run_training(
        train_loader, validation_loader
    )

    assert loaded_trainer.epoch == 4
    assert training_data.shape[0] == 1
    assert training_data["epoch"].tolist() == [3]
    assert len(validation_data) == 2


def test_snapshot_load_sets_missing_optional_fields_to_none(tmppath):
    snapshot_path = tmppath / "legacy" / "epoch_1"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": {"weight": torch.tensor([0.5])},
            "optimizer_state_dict": {"lr": 1e-3},
            "config_path": str(tmppath / "config.yaml"),
        },
        snapshot_path,
    )

    loaded = Snapshot.load(snapshot_path)

    assert loaded.lr_scheduler_state_dict is None
    assert loaded.validator_state_dict is None
    assert loaded.tester_state_dict is None
    assert loaded.early_stopping_state_dict is None


def test_snapshot_load_raises_for_missing_required_keys(tmppath):
    snapshot_path = tmppath / "broken" / "epoch_0"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": 0, "model_state_dict": {}, "config_path": "config.yaml"},
        snapshot_path,
    )

    with pytest.raises(ValueError, match="missing required keys"):
        Snapshot.load(snapshot_path)


def test_trainer_load_model(config, tmppath):
    trainer = QG.Trainer.from_config(config)

    assert trainer.model is not None

    original_weights = [param.clone() for param in trainer.model.parameters()]

    trainer.save_checkpoint("_model_snapshot")

    for param in trainer.model.parameters():
        param.data.zero_()

    loaded_model = trainer.load_model(
        trainer.checkpoint_path / "epoch_0_model_snapshot"
    )

    assert loaded_model is trainer.model
    assert isinstance(loaded_model, QG.GNNModel)

    for orig, loaded in zip(original_weights, loaded_model.parameters()):
        assert torch.all(torch.eq(orig, loaded.data))


def test_trainer_run_training(make_dataset, config):
    trainer = QG.Trainer.from_config(config)

    assert trainer.validator is not None
    assert trainer.model is not None

    test_loader, validation_loader, _ = make_loader_factory(config).prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )

    original_weights = [param.clone() for param in trainer.model.parameters()]

    training_data, valid_data = trainer.run_training(
        test_loader,
        validation_loader,
    )
    trained_weights = [param.clone() for param in trainer.model.parameters()]

    # Check if the model parameters have changed after training
    for orig, trained in zip(original_weights, trained_weights):
        assert not torch.all(torch.eq(orig, trained.data)), (
            "Model parameters did not change after training."
        )

    assert valid_data is not None  # has no validator
    assert len(valid_data) == config["training"]["num_epochs"]
    assert training_data.shape[0] == config["training"]["num_epochs"]
    assert len(trainer.validator.data) == config["training"]["num_epochs"]


def test_trainer_run_training_with_datasetconf(config_with_data):
    trainer = QG.Trainer.from_config(config_with_data)

    assert trainer.validator is not None
    assert trainer.model is not None

    train_loader, validation_loader, _ = make_loader_factory(
        config_with_data
    ).prepare_dataloaders(split=[0.8, 0.1, 0.1])

    original_weights = [param.clone() for param in trainer.model.parameters()]

    training_data, valid_data = trainer.run_training(
        train_loader,
        validation_loader,
    )
    trained_weights = [param.clone() for param in trainer.model.parameters()]

    # Check if the model parameters have changed after training
    for orig, trained in zip(original_weights, trained_weights):
        assert not torch.all(torch.eq(orig, trained.data)), (
            "Model parameters did not change after training."
        )

    assert valid_data is not None  # has no validator
    assert len(valid_data) == config_with_data["training"]["num_epochs"]
    assert training_data.shape[0] == config_with_data["training"]["num_epochs"]
    assert len(trainer.validator.data) == config_with_data["training"]["num_epochs"]


def test_trainer_run_training_with_default_evaluators(
    make_dataset, config_with_default_evaluators
):
    trainer = QG.Trainer.from_config(config_with_default_evaluators)

    assert isinstance(trainer.validator, QG.Validator)
    assert trainer.model is not None

    test_loader, validation_loader, _ = make_loader_factory(
        config_with_default_evaluators
    ).prepare_dataloaders(make_dataset, split=[0.8, 0.1, 0.1])

    original_weights = [param.clone() for param in trainer.model.parameters()]

    training_data, valid_data = trainer.run_training(
        test_loader,
        validation_loader,
    )
    trained_weights = [param.clone() for param in trainer.model.parameters()]

    for orig, trained in zip(original_weights, trained_weights):
        assert not torch.all(torch.eq(orig, trained.data))

    assert (
        len(trainer.validator.data)
        == config_with_default_evaluators["training"]["num_epochs"]
    )


def test_trainer_run_training_with_scheduler(make_dataset, config_with_scheduler):
    trainer = QG.Trainer.from_config(config_with_scheduler)

    assert trainer.validator is not None
    assert trainer.model is not None
    assert trainer.lr_scheduler is not None

    train_loader, _, __ = make_loader_factory(
        config_with_scheduler
    ).prepare_dataloaders(split=[0.8, 0.1, 0.1])

    for epoch in range(0, 3):
        original_weights = [param.clone() for param in trainer.model.parameters()]
        old_lr = trainer.lr_scheduler.get_last_lr()[0]
        trainer._run_train_epoch(trainer.model, trainer.optimizer, train_loader)

        trained_weights = [param.clone() for param in trainer.model.parameters()]

        # Check if the model parameters have changed after training
        for orig, trained in zip(original_weights, trained_weights):
            assert not torch.all(torch.eq(orig, trained.data)), (
                "Model parameters did not change after training."
            )
        assert trainer.lr_scheduler.get_last_lr()[0] != old_lr


def test_trainer_run_training_without_validator_returns_empty_validation_data(
    make_dataset, config, tmppath
):
    trainer = make_injected_trainer(
        config,
        tmppath / "no_validator",
        validator=None,
        tester=DummyEvaluator(),
        early_stopper=None,
    )

    train_loader, validation_loader, _ = make_loader_factory(
        config
    ).prepare_dataloaders(make_dataset, split=[0.8, 0.1, 0.1])

    training_data, validation_data = trainer.run_training(
        train_loader, validation_loader
    )

    assert training_data.shape[0] == config["training"]["num_epochs"]
    assert validation_data.empty


def test_trainer_run_test_without_tester_returns_empty_dict(
    make_dataset, config, tmppath
):
    trainer = make_injected_trainer(
        config,
        tmppath / "no_tester",
        validator=DummyEvaluator(),
        tester=None,
        early_stopper=None,
    )

    test_loader, _, _ = make_loader_factory(config).prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )

    test_data = trainer.run_test(test_loader)

    assert test_data == {}
