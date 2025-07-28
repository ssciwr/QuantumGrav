# refer to https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py

"""
Optuna example with Ray Tune that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd() + "/data"  # Directory to save FashionMNIST dataset.
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
MAX_LAYERS = 3  # Maximum number of layers to optimize


def define_search_space():
    """Define the search space for hyperparameters."""

    search_space = {
        "n_layers": tune.randint(1, MAX_LAYERS + 1),  # 1 to 3 layers
        "optimizer": tune.choice(["Adam", "RMSprop", "SGD"]),
        "lr": tune.loguniform(1e-5, 1e-1),
    }

    # Dynamically add Tune distributions based on n_layers
    for i in range(MAX_LAYERS):
        search_space[f"n_units_l{i}"] = tune.sample_from(
            lambda config, idx=i: (
                tune.randint(4, 129).sample(config)
                if idx < config["n_layers"]
                else None
            )
        )
        search_space[f"dropout_l{i}"] = tune.sample_from(
            lambda config, idx=i: (
                tune.uniform(0.2, 0.51).sample(config)
                if idx < config["n_layers"]
                else None
            )
        )

    return search_space


def define_algorithm():
    return OptunaSearch()


def define_model(config):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = config["n_layers"]
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = config[f"n_units_l{i}"]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = config[f"dropout_l{i}"]
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            DIR, train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(config):
    # Generate the model.
    model = define_model(config).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = config["optimizer"]
    lr = config["lr"]
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        tune.report(
            {
                "accuracy": accuracy,
                "epoch": epoch,
            }
        )

    return accuracy


if __name__ == "__main__":
    search_space = define_search_space()
    algorithm = define_algorithm()

    scheduler = ASHAScheduler(
        max_t=EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            num_samples=10,
            search_alg=algorithm,
            scheduler=scheduler,
            metric="accuracy",
            mode="max",
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name="raytune_pytorch_test",
            storage_path=os.getcwd() + "/ray_results",
            stop={"training_iteration": EPOCHS},
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="accuracy", mode="max")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")
    print(
        f"Best trial final training iteration: {best_result.metrics['training_iteration']}"
    )
    print(f"Best trial final epoch: {best_result.metrics['epoch']}")
