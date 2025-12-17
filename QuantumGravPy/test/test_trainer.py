import pytest
import torch
import torch_geometric
from torch_geometric.data import Data

import QuantumGrav as QG
import numpy as np
from functools import partial
import jsonschema
from pathlib import Path
import re
import datetime
import pandas as pd
from copy import deepcopy

torch.multiprocessing.set_start_method("spawn", force=True)  # for dataloader

# data transform functions


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
    cfg["training"]["lr_scheduler_kwargs"] = {
        "total_iters": 10,
        "last_epoch": -1 
    }
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


# this is needed for testing the full training loop
class DummyEvaluator:
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


def test_trainer_creation_works(config):
    trainer = QG.Trainer(
        config,
    )

    assert trainer.config == config
    assert trainer.criterion is compute_loss
    assert trainer.apply_model is None
    assert isinstance(trainer.early_stopping, QG.DefaultEarlyStopping)
    assert isinstance(trainer.validator, DummyEvaluator)
    assert isinstance(trainer.tester, DummyEvaluator)

    assert trainer.device == torch.device("cpu")
    assert trainer.seed == config["training"]["seed"]
    assert trainer.best_score is None
    assert trainer.best_epoch == 0
    assert trainer.epoch == 0
    assert trainer.checkpoint_at == config["training"].get("checkpoint_at", None)

    assert trainer.optimizer is None
    assert trainer.model is None


def test_trainer_model_instantiation_works(config):
    trainer = QG.Trainer(
        config,
    )
    assert trainer.model is None
    trainer.initialize_model()

    assert isinstance(trainer.model, QG.GNNModel)

def test_trainer_creation_broken(broken_config):
    with pytest.raises(
        jsonschema.ValidationError,
        match="'validation' is a required property",
    ):
        QG.Trainer(
            broken_config,
        )

def test_trainer_init_model(config):
    trainer = QG.Trainer(
        config,
    )
    model = trainer.initialize_model()
    assert model is not None
    assert isinstance(model, QG.GNNModel)


def test_trainer_init_optimizer(config):
    trainer = QG.Trainer(
        config,
    )
    # need a model to initialize the optimizer
    assert trainer.model is None
    trainer.initialize_model()
    assert isinstance(trainer.model, QG.GNNModel)

    assert trainer.optimizer is None
    trainer.initialize_optimizer()
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)

def test_trainer_init_optimizer_fails(config):
    trainer = QG.Trainer(
        config,
    )
    # need a model to initialize the optimizer
    assert trainer.model is None
    with pytest.raises(Exception, match="Model must be initialized before initializing optimizer."):
        trainer.initialize_optimizer()

def test_trainer_init_lr_scheduler(config_with_scheduler): 
    trainer = QG.Trainer(config_with_scheduler)
    trainer.initialize_model()
    trainer.initialize_optimizer()
    
    assert hasattr(trainer, "lr_scheduler") is False
    trainer.initialize_lr_scheduler()
    assert hasattr(trainer, "lr_scheduler") is True
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.LinearLR)
    assert trainer.lr_scheduler.start_factor == 0.2
    assert trainer.lr_scheduler.end_factor == 0.8
    assert trainer.lr_scheduler.total_iters == 10
    assert trainer.lr_scheduler.last_epoch == 0
    
def test_trainer_init_lr_scheduler_fails(config_with_scheduler): 
    trainer = QG.Trainer(config_with_scheduler)
    trainer.initialize_model()
    with pytest.raises(RuntimeError, match="Optimizer must be initialized before initializing learning rate scheduler."):
        trainer.initialize_lr_scheduler()
    assert hasattr(QG.Trainer, "lr_scheduler") is False

def test_trainer_prepare_dataloader(make_dataset, config):
    trainer = QG.Trainer(
        config,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
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
        trainer = QG.Trainer(
            config,
        )
        trainer.prepare_dataloaders(make_dataset)

    with pytest.raises(ValueError, match=re.escape("validation size cannot be 0")):
        config["data"]["split"] = [0.95, 0.01, 0.04]
        trainer = QG.Trainer(
            config,
        )
        trainer.prepare_dataloaders(make_dataset)

    with pytest.raises(ValueError, match=re.escape("test size cannot be 0")):
        config["data"]["split"] = [0.85, 0.14, 0.01]
        trainer = QG.Trainer(
            config,
        )
        trainer.prepare_dataloaders(
            make_dataset,
        )


def test_trainer_prepare_dataloader_with_dataconf(config_with_data):
    trainer = QG.Trainer(
        config_with_data,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
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
    trainer = QG.Trainer(
        config_with_data,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
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
    trainer = QG.Trainer(
        config_with_data,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
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


def test_trainer_train_epoch(make_dataset, config):
    trainer = QG.Trainer(
        config,
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()
    assert trainer.model is not None
    assert trainer.optimizer is not None

    train_loader, _, _ = trainer.prepare_dataloaders(make_dataset)
    trainer.model.train()

    eval_data = trainer._run_train_epoch(trainer.model, trainer.optimizer, train_loader)

    assert trainer.model.training is True
    assert len(eval_data) == len(train_loader)


def test_trainer_check_model_status(config):
    trainer = QG.Trainer(
        config,
    )

    trainer.initialize_model()

    loss = np.random.rand(10).tolist()
    other__loss = np.random.rand(10).tolist()
    data = pd.DataFrame({"loss": loss, "other_loss": other__loss})

    trainer.epoch = 1
    saved = trainer._check_model_status(data)
    assert saved is False

    # returns true when early stopping is triggered
    trainer.early_stopping = lambda x: True

    saved = trainer._check_model_status(data)

    assert saved is True

    partial_path = datetime.datetime.now().strftime("%Y-%m-%d_")
    paths = [
        f
        for f in list(Path(config["training"]["path"]).iterdir())
        if partial_path in f.name
    ]
    assert len(paths) == 1

    file_content = [f.name for f in paths[0].iterdir()]
    assert "config.yaml" in file_content
    assert "model_checkpoints" in file_content


def test_trainer_load_checkpoint(config):
    trainer = QG.Trainer(
        config,
    )

    trainer.initialize_model()

    assert trainer.model is not None

    trainer.save_checkpoint("test")

    original_weights = [param.clone() for param in trainer.model.parameters()]

    # set all the params to zero
    for param in trainer.model.parameters():
        param.data.zero_()

    # Load the checkpoint
    trainer.load_checkpoint("test")

    # Check if the model parameters are restored
    assert trainer.epoch == 0

    for orig, loaded in zip(original_weights, trainer.model.parameters()):
        assert torch.all(torch.eq(orig, loaded.data))

    assert trainer.latest_checkpoint is not None


def test_trainer_load_checkpoint_fails(config):
    "Test loading a checkpoint that does not exist or when model is none"
    trainer = QG.Trainer(
        config,
    )

    trainer.initialize_model()

    assert trainer.model is not None

    trainer.save_checkpoint("test")
    with pytest.raises(FileNotFoundError, match="Checkpoint file .* does not exist."):
        trainer.load_checkpoint("non_existent")

    trainer.model = None
    with pytest.raises(
        RuntimeError, match="Model must be initialized before loading checkpoint."
    ):
        trainer.load_checkpoint("test")


def test_trainer_run_training(make_dataset, config):
    trainer = QG.Trainer(
        config,
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()

    assert trainer.validator is not None
    assert trainer.model is not None

    test_loader, validation_loader, _ = trainer.prepare_dataloaders(
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
    trainer = QG.Trainer(
        config_with_data,
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()

    assert trainer.validator is not None
    assert trainer.model is not None

    train_loader, validation_loader, _ = trainer.prepare_dataloaders(
        split=[0.8, 0.1, 0.1]
    )

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


def test_trainer_run_training_with_scheduler(make_dataset, config_with_scheduler):
    trainer = QG.Trainer(config_with_scheduler)
    trainer.initialize_model()
    trainer.initialize_optimizer()
    trainer.initialize_lr_scheduler()
    
    assert trainer.validator is not None
    assert trainer.model is not None
    assert trainer.lr_scheduler is not None
    
    train_loader, _, __ = trainer.prepare_dataloaders(
        split=[0.8, 0.1, 0.1]
    )
    
    for epoch in range(0, 3):
        original_weights = [param.clone() for param in trainer.model.parameters()]
        old_lr = trainer.lr_scheduler.get_last_lr()[0]
        trainer._run_train_epoch(
            trainer.model, 
            trainer.optimizer, 
            train_loader
        )

        trained_weights = [param.clone() for param in trainer.model.parameters()]

        # Check if the model parameters have changed after training
        for orig, trained in zip(original_weights, trained_weights):
            assert not torch.all(torch.eq(orig, trained.data)), (
                "Model parameters did not change after training."
            )
        assert trainer.lr_scheduler.get_last_lr()[0] != old_lr

def test_trainer_run_test(make_dataset, config):
    trainer = QG.Trainer(
        config,
    )
    assert trainer.tester is not None
    trainer.initialize_model()
    trainer.initialize_optimizer()

    test_loader, _, _ = trainer.prepare_dataloaders(make_dataset, split=[0.8, 0.1, 0.1])

    trainer.save_checkpoint("best")  # needed or test will fail

    test_data = trainer.run_test(test_loader, "best")

    assert test_data is not None
    assert len(test_data) == 1  # DummyEvaluator returns a single loss value
    assert len(trainer.tester.data) == 1
