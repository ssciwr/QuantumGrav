import pytest
import torch


from torch_geometric.data import Data
import QuantumGrav as QG
import numpy as np
from pathlib import Path
import re
from datetime import datetime

torch.multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture
def tmppath(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoints")
    return path


@pytest.fixture
def config(model_config_eval, tmppath):
    cfg = {
        "training": {
            "seed": 42,
            # training loop
            "device": "cpu",
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
    }

    cfg["model"]["name"] = "GNNModel"

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


def test_trainer_creation_works(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    assert trainer.config == config
    assert trainer.criterion is compute_loss
    assert trainer.apply_model is not None
    assert trainer.early_stopping is None
    assert trainer.validator is None
    assert trainer.tester is None

    assert trainer.device == torch.device("cpu")
    assert trainer.seed == config["training"]["seed"]
    assert trainer.best_score is None
    assert trainer.best_epoch == 0
    assert trainer.epoch == 0
    assert trainer.checkpoint_at == config["training"].get("checkpoint_at", None)

    assert trainer.optimizer is None
    assert trainer.model is None


def test_trainer_creation_broken(broken_config):
    with pytest.raises(
        ValueError,
        match="Configuration must contain 'training', 'model', 'validation' and 'testing' sections.",
    ):
        QG.Trainer(
            broken_config,
            compute_loss,
            apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
            early_stopping=None,
            validator=None,
            tester=None,
        )


def test_trainer_init_model(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )
    model = trainer.initialize_model()
    assert model is not None
    assert isinstance(model, QG.GNNModel)


def test_trainer_init_optimizer(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )
    model = trainer.initialize_model()
    assert model is not None
    assert isinstance(model, QG.GNNModel)

    optimizer = trainer.initialize_optimizer()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_trainer_prepare_dataloader(make_dataset, config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    train_loader, val_loader, test_loader = trainer.prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )

    assert len(train_loader) == 3
    assert len(val_loader) == 1
    assert len(test_loader) == 2

    for batch in train_loader:
        assert isinstance(batch, Data)
        assert batch.x.shape == (60, 2)

    for batch in val_loader:
        assert isinstance(batch, Data)
        assert batch.x.shape == (15, 2)

    for batch in test_loader:
        assert isinstance(batch, Data)
        assert batch.x.shape == (15, 2)


def test_trainer_prepare_dataloader_broken(make_dataset, config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Split ratios must sum to 1.0. Provided split: [0.9, 0.2, 0.1]"
        ),
    ):
        trainer.prepare_dataloaders(make_dataset, split=[0.9, 0.2, 0.1])


def test_trainer_train_epoch(make_dataset, config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()

    train_loader, _, _ = trainer.prepare_dataloaders(
        make_dataset, split=[0.8, 0.1, 0.1]
    )
    trainer.model.train()

    eval_data = trainer._run_train_epoch(trainer.model, trainer.optimizer, train_loader)

    assert trainer.model.training is True
    assert len(eval_data) == len(train_loader)


def test_trainer_check_model_status(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=None,
        tester=None,
    )

    trainer.initialize_model()

    loss = np.random.rand(10).tolist()

    trainer.epoch = 1
    saved = trainer._check_model_status(loss)
    assert saved is False

    trainer.early_stopping = lambda x: True
    loss = np.random.rand(10).tolist()
    saved = trainer._check_model_status(loss)

    assert saved is True

    partial_path = datetime.now().strftime("%Y-%m-%d_")
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
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=None,
    )

    trainer.initialize_model()
    trainer.save_checkpoint()

    original_weights = [param.clone() for param in trainer.model.parameters()]

    # set all the params to zero
    for param in trainer.model.parameters():
        param.data.zero_()

    # Load the checkpoint
    trainer.load_checkpoint(0)

    # Check if the model parameters are restored
    for orig, loaded in zip(original_weights, trainer.model.parameters()):
        assert torch.all(torch.eq(orig, loaded.data))
    assert trainer.epoch == 0


# there is no test for the working 'save_checkpoint' method, as it is tested in the _check_model_status method above


def test_trainer_check_model_status_no_model(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=None,
        tester=None,
    )

    with pytest.raises(
        ValueError,
        match="Model must be initialized before saving checkpoint.",
    ):
        trainer.save_checkpoint()


def test_trainer_check_model_status_no_modelname(config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=None,
        tester=None,
    )
    trainer.initialize_model()
    del trainer.config["model"]["name"]

    with pytest.raises(
        ValueError,
        match="Model configuration must contain 'name' to save checkpoint.",
    ):
        trainer.save_checkpoint()


def test_trainer_run_training(make_dataset, config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=lambda x: False,
        validator=DummyEvaluator(),  # type: ignore
        tester=None,
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()

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
    assert len(training_data) == config["training"]["num_epochs"]
    assert len(trainer.validator.data) == config["training"]["num_epochs"]


def test_trainer_run_test(make_dataset, config):
    trainer = QG.Trainer(
        config,
        compute_loss,
        apply_model=lambda model, data: model(data.x, data.edge_index, data.batch),
        early_stopping=None,
        validator=None,
        tester=DummyEvaluator(),  # type: ignore
    )
    trainer.initialize_model()
    trainer.initialize_optimizer()

    test_loader, _, _ = trainer.prepare_dataloaders(make_dataset, split=[0.8, 0.1, 0.1])

    test_data = trainer.run_test(test_loader)

    assert test_data is not None
    assert len(test_data) == 1  # DummyEvaluator returns a single loss value
    assert len(trainer.tester.data) == 1
