import QuantumGrav as QG
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
import pytest


@pytest.fixture
def make_dataloader(create_data_hdf5, read_data):
    datadir, datafiles = create_data_hdf5

    dataset = QG.QGDataset(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=4,
        transform=lambda x: x,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,  # Ensure all batches are of the same size. last batches that are bad need to be handled by hand
    )
    return dataloader


@pytest.fixture
def model_config_eval():
    return {
        "gcn_net": [
            {
                "in_dim": 2,
                "out_dim": 8,
                "dropout": 0.3,
                "gnn_layer_type": "gcn",
                "normalizer": "batch_norm",
                "activation": "relu",
                "norm_args": [
                    8,
                ],
                "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
                "gnn_layer_kwargs": {
                    "cached": False,
                    "bias": True,
                    "add_self_loops": True,
                },
            },
            {
                "in_dim": 8,
                "out_dim": 12,
                "dropout": 0.3,
                "gnn_layer_type": "gcn",
                "normalizer": "batch_norm",
                "activation": "relu",
                "norm_args": [
                    12,
                ],
                "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
                "gnn_layer_kwargs": {
                    "cached": False,
                    "bias": True,
                    "add_self_loops": True,
                },
            },
        ],
        "classifier": {
            "input_dim": 12,
            "output_dims": [
                3,
            ],
            "hidden_dims": [24, 16],
            "activation": "relu",
            "backbone_kwargs": [{}, {}],
            "output_kwargs": [{}],
            "activation_kwargs": [{"inplace": False}],
        },
        "pooling_layer": "mean",
    }


@pytest.fixture
def gnn_model_eval(model_config_eval):
    """Fixture to create a GNNModel for evaluation."""
    model = QG.GNNModel.from_config(
        model_config_eval,
    )
    model.eval()
    return model


def compute_loss(x: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute the loss between predictions and targets."""
    loss = torch.nn.MSELoss()(x[0], data.y.to(torch.float32))
    return loss


def test_train_epoch(gnn_model_eval, make_dataloader):
    dataloader = make_dataloader
    optimizer = torch.optim.Adam(gnn_model_eval.parameters(), lr=0.01)
    loss_data = []
    initial_params = [p.clone() for p in gnn_model_eval.parameters()]
    assert gnn_model_eval.training is False, (
        "Model should be in eval mode before training."
    )
    gnn_model_eval.train()
    assert gnn_model_eval.training is True, (
        "Model should be in training mode after calling train()."
    )
    # Run a single training epoch
    QG.train_epoch(
        model=gnn_model_eval,
        data_loader=dataloader,
        optimizer=optimizer,
        criterion=compute_loss,
        loss_data=loss_data,
    )

    # check that the optimizer has updated the model parameters
    for initial_param, param in zip(initial_params, gnn_model_eval.parameters()):
        assert not torch.equal(initial_param, param), "Model parameters did not update."

    assert len(loss_data) > 0
    # if the names are in one element, they are in all
    assert "mean" in loss_data[0]
    assert "std" in loss_data[0]
    assert "min" in loss_data[0]
    assert "max" in loss_data[0]
    assert "median" in loss_data[0]
    assert "q25" in loss_data[0]
    assert "q75" in loss_data[0]

    assert all(
        all(
            loss_data[i][col] > 0.0
            for col in ["mean", "std", "min", "max", "median", "q25", "q75"]
        )
        for i in range(len(loss_data))
    ), "Loss data columns are not zero or negative."


def test_training(gnn_model_eval, make_dataloader):
    dataloader = make_dataloader
    optimizer = torch.optim.Adam(gnn_model_eval.parameters(), lr=0.01)
    loss_data = []

    gnn_model_eval.train()

    # Run a single training epoch
    for epoch in range(3):
        QG.train_epoch(
            model=gnn_model_eval,
            data_loader=dataloader,
            optimizer=optimizer,
            criterion=compute_loss,
            loss_data=loss_data,
        )

    assert loss_data[0]["mean"] > 0.0, (
        "Loss mean should be greater than zero after training."
    )
    assert loss_data[-1]["mean"] < loss_data[0]["mean"], (
        "Loss mean should decrease after training."
    )


def test_test_epoch(gnn_model_eval, make_dataloader):
    dataloader = make_dataloader
    loss_data = []

    assert gnn_model_eval.training is False, (
        "Model should be in eval mode before testing."
    )
    # Run a single testing epoch
    QG.test_epoch(
        model=gnn_model_eval,
        data_loader=dataloader,
        criterion=compute_loss,
        loss_data=loss_data,
    )
    assert gnn_model_eval.training is False, (
        "Model should be in eval mode after testing."
    )
    assert len(loss_data) > 0
    # if the names are in one element, they are in all
    assert "mean" in loss_data[0]
    assert "std" in loss_data[0]
    assert "min" in loss_data[0]
    assert "max" in loss_data[0]
    assert "median" in loss_data[0]
    assert "q25" in loss_data[0]
    assert "q75" in loss_data[0]

    assert all(
        all(
            loss_data[i][col] > 0.0
            for col in ["mean", "std", "min", "max", "median", "q25", "q75"]
        )
        for i in range(len(loss_data))
    ), "Loss data columns are not zero or negative."


# no test for validate epoch because it is the same as test_epoch
