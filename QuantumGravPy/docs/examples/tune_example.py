import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # make sure no GPU is used

from QGTune import tune
import optuna
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from optuna.storages.journal import JournalStorage, JournalFileBackend

DEVICE = torch.device("cpu")

current_dir = Path(__file__).parent
tmp_dir = current_dir / "tmp"


def create_tuning_config_file(tmp_path: Path) -> Path:
    """Create a tuning configuration YAML file for testing purposes."""
    # Note on n_jobs: Multi-thread optimization has traditionally been inefficient
    # in Python due to the Global Interpreter Lock (GIL) (Python < 3.14)
    tuning_config = {
        "config_file": str(tmp_path / "config.yaml"),
        "study_name": "test_study",
        "storage": str(tmp_path / "test_study.log"),
        "direction": "maximize",
        "n_trials": 15,
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


def create_config_file(file_path: Path) -> Path:
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
                values: [2, 5, 7]
    """

    with open(file_path, "w") as f:
        f.write(yaml_text)


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


def objective(trial, config_file: Path):

    search_space = tune.build_search_space(
        config_file=config_file,
        trial=trial,
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

    # training loop
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

        # report value to Optuna
        trial.report(accuracy, epoch)

        # prune trial if needed
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


def tune_integration(run_idx, tuning_config):  # run_idx is the iteration index

    study = tune.create_study(tuning_config)
    study.optimize(
        partial(objective, config_file=Path(tuning_config["config_file"])),
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

    return (
        len(study.trials),
        len(pruned_trials),
        len(complete_trials),
        study.best_trial.value,
    )


if __name__ == "__main__":
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # create tuning config
    tuning_config = tune.load_yaml(create_tuning_config_file(tmp_dir))
    n_processes = tuning_config.get("n_processes", 1)
    n_iterations = tuning_config.get("n_iterations", 1)

    # create config file
    create_config_file(Path(tuning_config["config_file"]))

    print("Starting the tuning integration process...")
    with Pool(processes=n_processes) as pool:
        local_results = pool.map(
            partial(tune_integration, tuning_config=tuning_config), range(n_iterations)
        )

    print("Tuning results for each run:------------------")
    for i, result in local_results:
        n_trials, n_pruned, n_complete, best_value = result
        print(f"Study statistics for run {i}: ")
        print("  Number of finished trials: ", n_trials)
        print("  Number of pruned trials: ", n_pruned)
        print("  Number of complete trials: ", n_complete)
        print("  Best trial value: ", best_value)

    storage = JournalStorage(
        JournalFileBackend(tuning_config["storage"])  # use uncompressed journal
    )
    study = optuna.load_study(study_name=tuning_config["study_name"], storage=storage)

    print("Best trial global:----------------------------")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("Saving the best configuration...")
    tune.save_best_config(
        config_file=Path(tuning_config["config_file"]),
        best_trial=trial,
        output_file=Path(tuning_config["best_config_file"]),
    )
