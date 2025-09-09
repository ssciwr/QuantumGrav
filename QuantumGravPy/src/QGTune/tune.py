import optuna.storages.journal
import yaml
from pathlib import Path
import optuna
from typing import Any, List
import copy
import sys


def _is_yaml_tuple_of_3(value: Any) -> bool:
    """Check if a value is a yaml tuple of 3 elements.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a tuple with 3 elements,
            False otherwise.
    """
    is_tuple_of_3 = (
        isinstance(value, dict)
        and value.get("type") == "tuple"
        and len(value.get("value", [])) == 3
    )
    return is_tuple_of_3


def _is_flat_list(value: Any) -> bool:
    """Check if a value is a flat list (not nested).

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a flat list, False otherwise.
    """
    return isinstance(value, list) and all(
        not isinstance(v, (list, dict)) for v in value
    )


def _is_suggest_categorical(value: Any) -> bool:
    """Check if a value should be an input for an Optuna suggest_categorical.
    The value would be a list of possible categories.
    E.g. ['relu', 'tanh', 'sigmoid'] or [16, 32, 64]

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a list, False otherwise.
    """
    non_empty_list = isinstance(value, list) and len(value) > 0
    valid_elements = (
        all(v is not None and isinstance(v, (bool, str, float, int)) for v in value)
        if non_empty_list
        else False
    ) and _is_flat_list(value)
    return non_empty_list and valid_elements


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
    is_tuple_of_3 = _is_yaml_tuple_of_3(value)
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
            All three are integers.
            False otherwise.
    """
    is_tuple_of_3 = _is_yaml_tuple_of_3(value)
    num_values = tuple(value.get("value", [])) if is_tuple_of_3 else ()
    all_ints = all(isinstance(v, int) for v in num_values) if is_tuple_of_3 else False
    return is_tuple_of_3 and all_ints


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
    if param_name.split(".")[-1] == "norm_args":
        # special case since norm_args can be a list of int
        return value

    if _is_suggest_categorical(value):
        return trial.suggest_categorical(param_name, value)
    elif _is_suggest_float(value):
        number_values = value.get("value", [])
        if isinstance(number_values[2], bool):
            return trial.suggest_float(
                param_name,
                float(number_values[0]),  # e.g. 1e-5
                float(number_values[1]),
                log=number_values[2],
            )
        else:
            return trial.suggest_float(
                param_name, number_values[0], number_values[1], step=number_values[2]
            )
    elif _is_suggest_int(value):
        number_values = value.get("value", [])
        return trial.suggest_int(
            param_name, number_values[0], number_values[1], step=number_values[2]
        )
    else:
        return value


def get_suggestion(
    config: dict, trial: optuna.trial.Trial, traced_param: list = []
) -> dict:
    """Get a dictionary of suggestions from a configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        trial (optuna.trial.Trial): The Optuna trial object.
        traced_param (list, optional): The list of traced parameters.

    Returns:
        dict: A dictionary of suggestions.
    """
    if isinstance(config, list):
        items = enumerate(config)
        suggestions = [None] * len(config)
    elif isinstance(config, dict):
        items = config.items()
        suggestions = {}
    else:
        return {}

    for param, value in items:
        traced_param.append(str(param))
        traced_name = ".".join(traced_param)
        suggestion = _convert_to_suggestion(traced_name, value, trial)
        if isinstance(suggestion, dict) or (
            isinstance(suggestion, list) and not _is_flat_list(suggestion)
        ):
            suggestions[param] = get_suggestion(suggestion, trial, traced_param)
        else:
            suggestions[param] = suggestion
        traced_param.pop()
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
        try:
            index = int(part)
            current = current[index]
        except ValueError:  # part is not an integer
            current = current.get(part)
    return current


def _walk_to_ref(node: Any, walk_path: List[str], config: dict) -> None:
    """A recursive function to walk to a reference in a
    dependency dictionary and resolve dependencies.

    Args:
        node (Any): The current node in the dependency dictionary.
        walk_path (List[str]): The path that has been walked so far.
        config (dict): The configuration dictionary.
    """
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return

    for key, value in items:
        if isinstance(value, (dict, list)):
            _walk_to_ref(value, walk_path + [key], config)
        else:
            resolved_value = _resolve_dependencies(config, value)
            parent = config
            for p in walk_path:
                parent = parent[p]
            # structure of depmap must map the structure of config
            parent[key] = resolved_value


def apply_dependencies(config: dict, depmap: dict) -> dict:
    """Apply dependencies from a dependency map to a configuration dictionary.

    Args:
        config (dict): The configuration dictionary.
        depmap (dict): The dependency map.
            E.g. {"model.gcn_net[0].in_dim": "data.num_node_features"}

    Returns:
        dict: The configuration dictionary with dependencies applied.
    """
    _walk_to_ref(depmap, [], config)
    return config


def load_yaml(file: Path | str, description: str) -> dict:
    """Load a YAML file, raising an error if it doesn't exist.

    Args:
        file (Path | str): Path to the YAML file.
        description (str): Description of the file for error messages.
    """
    if not file or not Path(str(file)).exists():
        raise FileNotFoundError(f"{description} file {file} does not exist.")
    with open(file, "r") as f:
        return yaml.safe_load(f)


def build_search_space_with_dependencies(
    search_space_file: Path,
    depmap_file: Path,
    trial: optuna.trial.Trial,
    tune_model: bool = False,
    tune_training: bool = True,
    base_settings_file: Path = None,
    built_search_space_file: Path = None,
) -> dict:
    """Build a hyperparameter search space from a YAML configuration file
    and a dependency map file, using an Optuna trial object.

    Args:
        search_space_file (Path): Path to the YAML configuration file.
        depmap_file (Path): Path to the dependency map YAML file.
        trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.
        tune_model (bool, optional): Whether to tune model hyperparameters.
            Defaults to False, meaning values for model will be taken from base settings.
        tune_training (bool, optional): Whether to tune training hyperparameters.
            Defaults to True, meaning values for training will be taken from search space.
        base_settings_file (Path, optional): Path to the base settings YAML file.
            This is required if either `tune_model` or `tune_training` is False.
            Defaults to None.
        built_search_space_file (Path, optional): Path to save the built search space
            with dependencies applied. If None, the built search space will not be saved.
            Defaults to None.

    Returns:
        dict: A dictionary representing the hyperparameter search space
            with dependencies applied.
    """
    search_space = load_yaml(search_space_file, description="Search space")
    depmap = load_yaml(depmap_file, description="Dependency map")
    base_settings = {}

    if not tune_model or not tune_training:
        base_settings = load_yaml(
            base_settings_file,
            description="Base settings file is required "
            "if you do not want to tune model or training. Base settings",
        )

        if not tune_model:
            search_space["model"] = base_settings["model"]
        if not tune_training:
            search_space["training"] = base_settings["training"]

    search_space_with_suggestions = get_suggestion(search_space, trial)
    search_space_with_deps = apply_dependencies(search_space_with_suggestions, depmap)

    if built_search_space_file:
        with open(built_search_space_file, "w") as f:
            yaml.safe_dump(search_space_with_deps, f)
    return search_space_with_deps


def get_tuning_settings(tuning_config_path: Path) -> dict:
    """Get hyperparameter tuning settings from a YAML configuration file.

    Args:
        tuning_config_path (Path): Path to the tuning configuration YAML file.

    Returns:
        dict: A dictionary containing the tuning settings.
    """
    return load_yaml(tuning_config_path, description="Tuning config")


def create_study(tuning_config: dict) -> None:
    """Create an Optuna study and save to a specified storage.

    Args:
        tuning_config (dict): The configuration dictionary for tuning settings.
    """
    storage = tuning_config.get("storage")
    study_name = tuning_config.get("study_name")
    direction = tuning_config.get("direction", "minimize")

    # create storage object if storage is provided
    optuna_storage = None
    if storage is not None:
        lock_obj = None

        if sys.platform.startswith("win"):
            lock_obj = optuna.storages.journal.JournalFileOpenLock(str(storage))

        backend = optuna.storages.journal.JournalFileBackend(
            str(storage), lock_obj=lock_obj
        )

        optuna_storage = optuna.storages.JournalStorage(backend)

    study = optuna.create_study(
        study_name=study_name,
        storage=optuna_storage,
        load_if_exists=True,
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=5, interval_steps=5
        ),
    )
    print(f"Study {study_name} was created and saved to {storage}.")
    return study


def save_best_trial(study: optuna.study.Study, output_path: Path) -> None:
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


def save_best_config(
    built_search_space_file: Path,
    best_trial_file: Path,
    depmap_file: Path,
    output_file: Path,
):
    """Save the best configuration by combining the best trial parameters
    with the built search space.

    Args:
        built_search_space_file (Path): Path to the built search space YAML file.
        best_trial_file (Path): Path to the best trial parameters YAML file.
            Keys in this file are joined keys from the built search space.
        depmap_file (Path): Path to the dependency map YAML file.
            This is necessary to resolve dependencies in the built search space.
            As the built search space was saved with first trial values.
        output_file (Path): Path to the output YAML file for the best configuration.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")

    built_search_space = load_yaml(
        built_search_space_file, description="Built search space"
    )
    best_trial = load_yaml(best_trial_file, description="Best trial")

    best_config = copy.deepcopy(built_search_space)

    def _get_next(current: dict, part: str | int):
        try:
            index = int(part)
            return current[index]
        except ValueError:
            return current.get(part)

    def _set_next(current: dict, part: str | int, value: Any):
        try:
            index = int(part)
            current[index] = value
        except ValueError:
            current[part] = value

    for key, value in best_trial.items():
        parts = key.split(".")
        current = best_config
        for part in parts[:-1]:
            current = _get_next(current, part)
        _set_next(current, parts[-1], value)

    # apply dependencies to resolve any changes
    depmap = load_yaml(depmap_file, description="Dependency map")
    best_config = apply_dependencies(best_config, depmap)

    with open(output_file, "w") as file:
        yaml.safe_dump(best_config, file)

    print(f"Best configuration saved to {output_file}.")
