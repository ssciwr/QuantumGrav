from QGTune import tune
import optuna
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path


DEVICE = torch.device("cpu")
current_dir = Path(__file__).parent
tmp_dir = current_dir / "tmp"


def get_config_file(tmp_path):
    config = {
        "model": {
            "n_layers": 3,
            "nn": [
                {
                    "in_dim": 784,
                    "out_dim": [128, 256],
                    "dropout": {"type": "tuple", "value": [0.2, 0.5, 0.1]},
                },
                {
                    "in_dim": "ref",
                    "out_dim": [128, 256],
                    "dropout": {"type": "tuple", "value": [0.2, 0.5, 0.1]},
                },
                {
                    "in_dim": "ref",
                    "out_dim": [16, 32],
                    "dropout": {"type": "tuple", "value": [0.2, 0.5, 0.1]},
                },
            ],
        },
        "training": {
            "batch_size": [16, 32],
            "optimizer": ["Adam", "SGD"],
            "lr": {"type": "tuple", "value": [1e-5, 1e-1, True]},
            "epochs": [2, 5],
        },
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    return config_file


def get_dependency_file(tmp_path):
    deps = {
        "model": {
            "nn": [
                {},
                {
                    "in_dim": "model.nn.0.out_dim",
                },
                {
                    "in_dim": "model.nn.1.out_dim",
                },
            ],
        }
    }
    dep_file = tmp_path / "deps.yaml"
    with open(dep_file, "w") as f:
        yaml.safe_dump(deps, f)
    return dep_file


def get_base_config_file(tmp_path):
    base_config = {
        "model": {
            "n_layers": 3,
            "nn": [
                {
                    "in_dim": 784,
                    "out_dim": 256,
                    "dropout": 0.2,
                },
                {
                    "in_dim": 256,
                    "out_dim": 256,
                    "dropout": 0.2,
                },
                {
                    "in_dim": 256,
                    "out_dim": 32,
                    "dropout": 0.2,
                },
            ],
        },
        "training": {"batch_size": 32, "optimizer": "Adam", "lr": 0.001, "epochs": 5},
    }
    base_config_file = tmp_path / "base_config.yaml"
    with open(base_config_file, "w") as f:
        yaml.safe_dump(base_config, f)

    return base_config_file


def define_small_model(config):
    n_layers = config["model"]["n_layers"]
    layers = []

    for i in range(n_layers):
        in_dim = config["model"]["nn"][i]["in_dim"]
        out_dim = config["model"]["nn"][i]["out_dim"]
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
        dropout = config["model"]["nn"][i]["dropout"]
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(out_dim, 10))  # classification of 10 digits
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def load_data(config, dir_path):
    batch_size = config["training"]["batch_size"]
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            dir_path, train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(dir_path, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial):
    search_space = tune.build_search_space_with_dependencies(
        get_config_file(tmp_dir),
        get_dependency_file(tmp_dir),
        trial,
        tune_model=True,
        tune_training=True,
        base_settings_file=get_base_config_file(tmp_dir),
        built_search_space_file=tmp_dir / "built_search_space.yaml",
    )

    # prepare model
    model = define_small_model(search_space).to(DEVICE)

    # prepare optimizer
    optimizer_name = search_space["training"]["optimizer"]
    lr = search_space["training"]["lr"]
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # prepare data
    data_dir = tmp_dir / "data"
    train_loader, valid_loader = load_data(search_space, data_dir)

    # train the model
    epochs = search_space["training"]["epochs"]
    batch_size = search_space["training"]["batch_size"]
    n_train_examples = batch_size * 30
    n_valid_examples = batch_size * 10

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * batch_size > n_train_examples:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.NLLLoss()(output, target)
            loss.backward()
            optimizer.step()

        # validate the model
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                if batch_idx * batch_size > n_valid_examples:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), n_valid_examples)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


def tune_integration():
    # Note: Multi-thread optimization has traditionally been inefficient in Python
    # due to the Global Interpreter Lock (GIL) (Python < 3.14)
    tuning_config = {
        "study_name": "test_study",
        "storage": None,
        "direction": "maximize",
        "n_trials": 10,
        "timeout": 600,
        "n_jobs": 1,  # set to >1 to enable multi-threading
    }

    study = tune.create_study(tuning_config)
    study.optimize(
        objective,
        n_trials=tuning_config["n_trials"],
        timeout=tuning_config["timeout"],
        n_jobs=tuning_config["n_jobs"],  # pass n_jobs to optimize method
    )

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("Save best trial to best_trial.yaml")
    tune.save_best_trial(study, tmp_dir / "best_trial.yaml")

    print("Save best config to best_config.yaml")
    tune.save_best_config(
        built_search_space_file=tmp_dir / "built_search_space.yaml",
        best_trial_file=tmp_dir / "best_trial.yaml",
        depmap_file=tmp_dir / "deps.yaml",
        output_file=tmp_dir / "best_config.yaml",
    )


if __name__ == "__main__":
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Starting the tuning integration process...")
    tune_integration()
