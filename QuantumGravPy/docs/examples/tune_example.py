from QGTune import tune
import optuna
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
from multiprocessing import Pool
from functools import partial


DEVICE = torch.device("cpu")
current_dir = Path(__file__).parent
tmp_dir = current_dir / "tmp"


def create_tuning_config_file(tmp_path: Path) -> Path:
    """Create a tuning configuration YAML file for testing purposes."""
    # Note on n_jobs: Multi-thread optimization has traditionally been inefficient
    # in Python due to the Global Interpreter Lock (GIL) (Python < 3.14)
    tuning_config = {
        "study_name": "test_study",
        "storage": str(tmp_path / "test_study.log"),
        "direction": "maximize",
        "n_trials": 10,
        "timeout": 600,
        "n_jobs": 1,  # set to >1 to enable multi-threading,
        "best_config_file": str(tmp_path / "best_config.yaml"),
        "n_processes": 2,
        "n_iterations": 2,
    }
    tuning_config_file = tmp_path / "tuning_config.yaml"
    with open(tuning_config_file, "w") as f:
        yaml.safe_dump(tuning_config, f)

    return tuning_config_file


def create_config_file(tmp_dir: Path) -> Path:
    """Create a base configuration YAML file with tuning parameters for testing purposes."""
    yaml_text = """
        model:
            n_layers: 3
            nn:
                -
                    in_dim: 784
                    out_dim: !sweep
                        values: [128, 256]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
                -
                    in_dim: !reference
                        target: model.nn[0].out_dim
                    out_dim: !sweep
                        values: [16, 32]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
                -
                    in_dim: !reference
                        target: model.nn[1].out_dim
                    out_dim: !sweep
                        values: [16, 32]
                    dropout: !range
                        start: 0.2
                        stop: 0.5
                        step: 0.1
        training:
            batch_size: !sweep
                values: [16, 32]
            optimizer: !sweep
                values: ["Adam", "SGD"]
            lr: !range
                start: 1e-5
                stop: 1e-2
                log: true
            epochs: !sweep
                values: [2, 5]
    """

    config_file = tmp_dir / "config.yaml"
    with open(config_file, "w") as f:
        f.write(yaml_text)

    return config_file


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

    layers.append(nn.Linear(out_dim, 10))  # classification of 10 classes
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


def objective(trial, tuning_config):
    config_file = get_base_config_file(tuning_config.get("base_settings_file"))
    search_space_file = get_search_space_file(tuning_config.get("search_space_file"))
    depmap_file = get_dependency_file(tuning_config.get("depmap_file"))
    tune_model = tuning_config.get("tune_model")
    tune_training = tuning_config.get("tune_training")
    built_search_space_file = tuning_config.get("built_search_space_file")

    search_space = tune.build_search_space_with_dependencies(
        search_space_file,
        depmap_file,
        trial,
        tune_model=tune_model,
        tune_training=tune_training,
        base_settings_file=base_config_file,
        built_search_space_file=built_search_space_file,
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


def tune_integration(_, tuning_config):  # _ is the iteration index

    study = tune.create_study(tuning_config)
    study.optimize(
        partial(objective, tuning_config=tuning_config),
        n_trials=tuning_config["n_trials"],
        timeout=tuning_config["timeout"],
        n_jobs=tuning_config["n_jobs"],
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

    tuning_config = tune.get_tuning_settings(get_tuning_config_file(tmp_dir))
    n_processes = tuning_config.get("n_processes", 1)
    n_iterations = tuning_config.get("n_iterations", 1)

    print("Starting the tuning integration process...")
    with Pool(processes=n_processes) as pool:
        pool.map(
            partial(tune_integration, tuning_config=tuning_config), range(n_iterations)
        )
