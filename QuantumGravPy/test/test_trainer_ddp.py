import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import QuantumGrav as QG
import numpy as np
from copy import deepcopy
import os
import h5py


@pytest.fixture
def tmppath(tmp_path_factory):
    checkpoint_path = tmp_path_factory.mktemp("checkpoints")
    return checkpoint_path


@pytest.fixture
def config(model_config_eval, tmppath):
    cfg = {
        "training": {
            "seed": 42,
            # training loop
            "device": "cpu",
            "early_stopping_patience": 10,
            "checkpoint_at": 20,
            "path": tmppath,
            # optimizer
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            # training loader
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": True,
            "num_epochs": 13,
            # "prefetch_factor": 2,
        },
        "model": model_config_eval,
        "validation": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": True,
        },
        "testing": {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": False,
        },
        "parallel": {
            "world_size": 1,
            "master_addr": "localhost",
            "master_port": "23456",
            "backend": "gloo",
        },
    }

    cfg["model"]["name"] = "GNNModel"

    return cfg


@pytest.fixture
def broken_config(config):
    cfg = deepcopy(config)
    del cfg["parallel"]

    return cfg


class DummyEvaluator:
    def __init__(self):
        self.data = []

    def validate(self, model, data_loader):
        # Dummy validation logic
        return [torch.rand(1)]

    def test(self, model, data_loader):
        # Dummy test logic
        return [torch.rand(1)]

    def report(self, losses: list):  # type: ignore
        avg = np.mean(losses)
        sigma = np.std(losses)
        print(f"Validation average loss: {avg}, Standard deviation: {sigma}")
        self.data.append((avg, sigma))


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0], data.y.to(torch.float32))
    return loss


def reader(f: h5py.File, idx: int, float_dtype, int_dtype, validate) -> Data:
    adj_raw = f["adjacency_matrix"][idx, :, :]
    adj_matrix = torch.tensor(adj_raw, dtype=float_dtype)
    edge_index, edge_weight = dense_to_sparse(adj_matrix)
    adj_matrix = adj_matrix.to_sparse()
    node_features = []

    # Path lengths
    max_path_future = torch.tensor(
        f["max_pathlen_future"][idx, :], dtype=float_dtype
    ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

    max_path_past = torch.tensor(
        f["max_pathlen_past"][idx, :], dtype=float_dtype
    ).unsqueeze(1)  # make this a (num_nodes, 1) tensor
    node_features.extend([max_path_future, max_path_past])

    x = torch.cat(node_features, dim=1)

    manifold = f["manifold"][idx]
    boundary = f["boundary"][idx]
    dimension = f["dimension"][idx]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(1),
        y=torch.tensor([[manifold, boundary, dimension]], dtype=int_dtype),
    )

    if validate and not data.validate():
        raise ValueError("Data validation failed.")
    return data


def test_initialize_ddp():
    QG.initialize_ddp(0, 1, "localhost", "23456", "gloo")
    assert os.environ["MASTER_ADDR"] == "localhost"
    assert os.environ["MASTER_PORT"] == "23456"
    assert torch.distributed.is_initialized(), (
        "Distributed process group was not initialized"
    )

    QG.cleanup_ddp()

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ
    assert torch.distributed.is_initialized() is False


def test_trainer_ddp_creation_works(config):
    trainer = QG.TrainerDDP(
        1,
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    assert trainer.rank == 1
    assert trainer.device == torch.device("cpu")
    assert trainer.world_size == 1


def test_trainer_ddp_creation_broken(broken_config):
    with pytest.raises(
        ValueError, match="Configuration must contain 'parallel' section for DDP."
    ):
        QG.TrainerDDP(
            1,
            broken_config,
            compute_loss,
            apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
            early_stopping=None,
            validator=None,
            tester=None,
        )


def test_trainer_ddp_init_model(config):
    QG.initialize_ddp(0, 1, "localhost", "23456", "gloo")

    trainer = QG.TrainerDDP(
        0,
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    trainer.initialize_model()
    assert trainer.model is not None

    QG.cleanup_ddp()


def test_trainer_ddp_prepare_dataloaders(make_dataset, config):
    QG.initialize_ddp(0, 1, "localhost", "23456", "gloo")

    trainer = QG.TrainerDDP(
        0,
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        make_dataset, split=[0.7, 0.15, 0.15]
    )

    assert isinstance(trainer.train_sampler, torch.utils.data.DistributedSampler)
    assert isinstance(trainer.val_sampler, torch.utils.data.DistributedSampler)
    assert isinstance(trainer.test_sampler, torch.utils.data.DistributedSampler)

    assert len(train_loader) == 2
    assert len(val_loader) == 2
    assert len(test_loader) == 3

    QG.cleanup_ddp()


def test_trainer_ddp_check_model_status(config, make_dataloader):
    QG.initialize_ddp(0, 1, "localhost", "23456", "gloo")

    trainer = QG.TrainerDDP(
        0,
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=DummyEvaluator(),  # type: ignore
        tester=DummyEvaluator(),  # type: ignore
    )
    trainer.initialize_model()
    trainer.epoch = 10
    trainer.checkpoint_at = 7
    eval_data = [0.1, 0.2, 0.3]
    saved = trainer._check_model_status(eval_data)

    assert saved is False

    trainer.checkpoint_at = 5j
    saved = trainer._check_model_status(eval_data)
    assert saved is True

    trainer.rank = 1
    saved = trainer._check_model_status(eval_data)
    assert saved is False

    QG.cleanup_ddp()


def test_trainer_ddp_run_training(config, make_dataset):
    QG.initialize_ddp(0, 1, "localhost", "23456", "gloo")
    trainer = QG.TrainerDDP(
        0,
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=DummyEvaluator(),  # type: ignore
        tester=DummyEvaluator(),  # type: ignore
    )

    train_loader, validation_loader, _ = trainer.prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )

    training_data, valid_data = trainer.run_training(train_loader, validation_loader)
    assert training_data is not None
    assert valid_data is not None
    assert len(valid_data) == 1
    assert len(training_data) == config["parallel"]["world_size"]
    assert len(valid_data) == config["parallel"]["world_size"]

    assert len(training_data[0]) == config["training"]["num_epochs"]
    assert len(trainer.validator.data) == config["training"]["num_epochs"]
    QG.cleanup_ddp()
