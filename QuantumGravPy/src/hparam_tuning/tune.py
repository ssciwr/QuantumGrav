import yaml
from pathlib import Path
import optuna
from typing import Any


def _is_suggest_caterorical(value: Any) -> bool:
    """Check if a value should be an input for an Optuna suggest_categorical.
    The value would be a list of possible categories.
    E.g. ['relu', 'tanh', 'sigmoid'] or [16, 32, 64]

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a list, False otherwise.
    """
    return isinstance(value, list)


def _is_suggest_float(value: Any) -> bool:
    """Check if a value should be an input for an Optuna suggest_float.
    The value would be a tuple of (min, max, step) or (min, max, log).
    E.g. (0.001, 0.1, 0.001) or (1e-5, 1e-1, True)

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a tuple with 3 elements,
            The first two numbers are floats,
            and the third is either a float or a bool.
            False otherwise.
    """
    is_tuple_of_3 = (
        isinstance(value, dict)
        and value.get("type") == "tuple"
        and len(value.get("value", [])) == 3
    )
    num_values = tuple(value.get("value", [])) if is_tuple_of_3 else ()
    two_floats = (
        (isinstance(num_values[0], float) and isinstance(num_values[1], float))
        if is_tuple_of_3
        else False
    )
    third_is_float_or_bool = (
        (isinstance(num_values[2], float) or isinstance(num_values[2], bool))
        if is_tuple_of_3
        else False
    )
    return is_tuple_of_3 and two_floats and third_is_float_or_bool


def _is_suggest_int(value: Any) -> bool:
    """Check if a value should be an input for an Optuna suggest_int.
    The value would be a tuple of (min, max, step).
    E.g. (16, 128, 16)

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a tuple with 3 elements,
            The first two numbers are integers,
            and the third is an integer.
            False otherwise.
    """
    is_tuple_of_3 = isinstance(value, tuple) and len(value) == 3
    two_ints = (
        (isinstance(value[0], int) and isinstance(value[1], int))
        if is_tuple_of_3
        else False
    )
    third_is_int = (isinstance(value[2], int)) if is_tuple_of_3 else False
    return is_tuple_of_3 and two_ints and third_is_int


def _convert_to_suggestion(
    param_name: str, value: Any, trial: optuna.trial.Trial
) -> Any:
    """Convert a value to an Optuna trial suggestion.

    Args:
        param_name (str): The name of the parameter.
        value (Any): The value to convert.
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        Any: The converted value.
    """
    if _is_suggest_caterorical(value):
        return trial.suggest_categorical(param_name, value)
    elif _is_suggest_float(value):
        if isinstance(value[2], bool):
            return trial.suggest_float(param_name, value[0], value[1], log=value[2])
        else:
            return trial.suggest_float(param_name, value[0], value[1], step=value[2])
    elif _is_suggest_int(value):
        return trial.suggest_int(param_name, value[0], value[1], step=value[2])
    else:
        return value


def get_suggestion(config: dict, trial: optuna.trial.Trial) -> dict:
    """Get a dictionary of suggestions from a configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        dict: A dictionary of suggestions.
    """
    suggestions = {}
    for param, value in config.items():
        suggestion = _convert_to_suggestion(param, value, trial)
        if isinstance(suggestion, dict):
            suggestions[param] = get_suggestion(suggestion, trial)
        else:
            suggestions[param] = suggestion
    return suggestions


def _resolve_dependencies(config: dict, ref_path: str) -> Any:
    """Resolve dependencies in a configuration.

    Args:
        config (dict): The configuration dictionary.
        ref_path (str): The reference path to the current position in the config.
            E.g. `gcn_net[0].out_dim`

    Returns:
        Any: The resolved value pointed by the reference path.
    """
    parts = ref_path.replace("]", "").replace("[", ".").split(".")
    current = config
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = current[part]
    return current


def apply_dependencies(config: dict, depmap: dict) -> dict:
    """Apply dependencies from a dependency map to a configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        depmap (dict): The dependency map.
            E.g. {"model.gcn_net[0].in_dim": "data.num_node_features"}

    Returns:
        dict: The configuration dictionary with dependencies applied.
    """
    for part, deps in depmap.items():
        for key, layers in deps.items():
            for i, layer_dep in enumerate(layers):
                for field, ref in layer_dep.items():
                    resolved_value = _resolve_dependencies(config, ref)
                    config[part][key][i][field] = (
                        [resolved_value] if field == "norm_args" else resolved_value
                    )
    return config


def build_search_space_with_dependencies(
    config_file: Path, depmap_file: Path, trial: optuna.trial.Trial
) -> dict:
    """Build a hyperparameter search space from a YAML configuration file
    and a dependency map file, using an Optuna trial object.

    Args:
        config_file (Path): Path to the YAML configuration file.
        depmap_file (Path): Path to the dependency map YAML file.
        trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.

    Returns:
        dict: A dictionary representing the hyperparameter search space
            with dependencies applied.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    if not depmap_file.exists():
        raise FileNotFoundError(f"Dependency map file {depmap_file} does not exist.")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    with open(depmap_file, "r") as file:
        depmap = yaml.safe_load(file)

    search_space = get_suggestion(config, trial)
    search_space_with_deps = apply_dependencies(search_space, depmap)
    return search_space_with_deps


def get_tunning_settings(tunning_config_path: Path) -> dict:
    """Get hyperparameter tuning settings from a YAML configuration file.

    Args:
        tunning_config_path (Path): Path to the tuning configuration YAML file.

    Returns:
        dict: A dictionary containing the tuning settings.
    """
    if not tunning_config_path.exists():
        raise FileNotFoundError(
            f"Tunning config file {tunning_config_path} does not exist."
        )

    with open(tunning_config_path, "r") as file:
        tunning_config = yaml.safe_load(file)

    return tunning_config


def create_study(tunning_config: dict) -> None:
    """Create an Optuna study and save to a specified storage.

    Args:
        tunning_config (dict): The configuration dictionary for tuning settings.
    """
    storage = tunning_config.get("storage")
    study_name = tunning_config.get("study_name")
    direction = tunning_config.get("direction", "minimize")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=5, interval_steps=5
        ),
    )
    print(f"Study {study_name} was created and saved to {storage}.")
    return study


def save_best_trail(study: optuna.study.Study, output_path: Path) -> None:
    """Save the best trial of an Optuna study to a YAML file.

    Args:
        study (optuna.study.Study): The Optuna study object.
        output_path (Path): Path to the output YAML file.
    """
    best_trial = study.best_trial
    best_params = best_trial.params

    with open(output_path, "w") as file:
        yaml.safe_dump(best_params, file)

    print(f"Best trial parameters saved to {output_path}.")
