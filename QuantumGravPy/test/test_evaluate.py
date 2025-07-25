import QuantumGrav as QG
from torch_geometric.data import DataLoader
import torch
import pytest
import pandas as pd


@pytest.fixture
def make_dataloader(create_data, read_data):
    datadir, datafiles = create_data

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
        batch_size=2,
        shuffle=True,
    )
    return dataloader


def test_train_epoch(gnn_model, make_dataloader):
    dataloader = make_dataloader
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    loss_data = pd.DataFrame()

    initial_params = [p.clone() for p in gnn_model.parameters()]

    # Run a single training epoch
    QG.train_epoch(
        model=gnn_model,
        data_loader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        loss_data=loss_data,
    )

    assert not loss_data.empty
    assert "mean" in loss_data.columns
    assert "std" in loss_data.columns
    assert "min" in loss_data.columns
    assert "max" in loss_data.columns
    assert "median" in loss_data.columns
    assert "q25" in loss_data.columns
    assert "q75" in loss_data.columns

    # check that the optimizer has updated the model parameters
    for initial_param, param in zip(initial_params, gnn_model.parameters()):
        assert not torch.equal(initial_param, param), "Model parameters did not update."

    assert all(
        all(loss_data[col, :] > 0.0)
        for col in ["mean", "std", "min", "max", "median", "q25", "q75"]
    ), "Loss data columns are not zero or negative."


def test_test_epoch(gnn_model, make_dataloader):
    dataloader = make_dataloader
    criterion = torch.nn.MSELoss()
    loss_data = pd.DataFrame()

    # Run a single testing epoch
    QG.test_epoch(
        model=gnn_model,
        data_loader=dataloader,
        criterion=criterion,
        loss_data=loss_data,
    )

    assert not loss_data.empty
    assert "mean" in loss_data.columns
    assert "std" in loss_data.columns
    assert "min" in loss_data.columns
    assert "max" in loss_data.columns
    assert "median" in loss_data.columns
    assert "q25" in loss_data.columns
    assert "q75" in loss_data.columns

    assert all(
        all(loss_data[col, :] > 0.0)
        for col in ["mean", "std", "min", "max", "median", "q25", "q75"]
    ), "Loss data columns are not zero or negative."


# no test for validate epoch because it is the same as test_epoch
