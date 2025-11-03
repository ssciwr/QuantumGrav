import optuna.storages.journal
import yaml
from pathlib import Path
import optuna
from typing import Any, List, Dict, Tuple
import copy
import sys
import QuantumGrav as QG


def is_flat_list(value: Any) -> bool:
    """Check if a value is a flat list (not nested).

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a flat list, False otherwise.
    """
    return isinstance(value, list) and all(
        not isinstance(v, (list, dict)) for v in value
    )


def is_categorical_suggestion(value: Any) -> bool:
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
    ) and is_flat_list(value)
    return non_empty_list and valid_elements


def is_float_suggestion(value: Any) -> bool:
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
    is_tuple_of_3 = len(value) == 3
    num_values = tuple(value) if is_tuple_of_3 else ()
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


def is_int_suggestion(value: Any) -> bool:
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
    is_tuple_of_3 = len(value) == 3
    num_values = tuple(value) if is_tuple_of_3 else ()
    all_ints = all(isinstance(v, int) for v in num_values) if is_tuple_of_3 else False
    return is_tuple_of_3 and all_ints


def get_value_of_ref(config: Dict[str, Any], ref_path: str) -> Any:
    """Get the value pointed by a reference path in the configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        ref_path (str): The reference path to a position in the config.
            E.g. `gcn_net[0].out_dim`

    Returns:
        Any: The value pointed by the reference path.
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


def convert_to_suggestion(
    param_name: str,
    node: Dict[str, Any],
    trial: optuna.trial.Trial,
    config: Dict[str, Any],
) -> Any:
    """Convert a value to an Optuna trial suggestion.

    Args:
        param_name (str): The name of the parameter.
        node (Dict[str, Any]): The YAML node containing the value.
        trial (optuna.trial.Trial): The Optuna trial object.
        config (Dict[str, Any]): The configuration dictionary for resolving references.

    Returns:
        Any: The converted value.
    """
    node_type = node.get("type")
    is_sweep = node_type == "sweep"
    is_coupled_sweep = node_type == "coupled-sweep"
    is_range = node_type == "range"

    if is_range:
        node_values = node.get("tune_values")
    else:
        node_values = node.get("values")

    if is_sweep and is_categorical_suggestion(node_values):
        return trial.suggest_categorical(param_name, node_values)
    elif is_range and is_float_suggestion(node_values):
        start, stop, step_or_log = node_values
        if isinstance(step_or_log, bool):
            return trial.suggest_float(
                param_name,
                float(start),  # e.g. 1e-5
                float(stop),
                log=step_or_log,
            )
        else:
            return trial.suggest_float(param_name, start, stop, step=step_or_log)
    elif is_range and is_int_suggestion(node_values):
        return trial.suggest_int(
            param_name, node_values[0], node_values[1], step=node_values[2]
        )
    elif is_coupled_sweep:
        # return dictionary mapping target values and coupled values
        target_values = get_value_of_ref(config, node.get("target"))
        coupled_values = node.get("values")
        if len(target_values) != len(coupled_values):
            raise ValueError(
                f"Length of target values of {node.get("target")} "
                f"and coupled values of {param_name} do not match."
            )
        mapped_values = dict(zip(target_values, coupled_values))
        coupled_sweep = {
            "type": "coupled-sweep-mapping",
            "target": node.get(
                "target"
            ),  # keep target to map with Optuna suggestions in later steps
            "mapping": mapped_values,
        }
        return coupled_sweep
    else:
        return node  # return the original value for deeper processing


def get_suggestion(
    config: Dict[str, Any],
    current_node: Dict[str, Any] | List,
    trial: optuna.trial.Trial,
    traced_param: list = [],
    coupled_sweep_mapping: Dict[str, Any] = {},
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get a dictionary of suggestions from a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        current_node (Dict[str, Any] | List): The current configuration node.
        trial (optuna.trial.Trial): The Optuna trial object.
        traced_param (list, optional): The list of traced parameters.
        coupled_sweep_mapping (Dict[str, Any], optional): The mapping for coupled sweeps.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - A dictionary of suggestions.
            - A dictionary mapping coupled sweep targets to their values.
    """
    if isinstance(current_node, list):
        items = enumerate(current_node)
        suggestions = [None] * len(current_node)
    elif isinstance(current_node, dict):
        items = current_node.items()
        suggestions = {}
    else:
        return {}

    for param, value in items:
        traced_param.append(str(param))
        traced_name = ".".join(traced_param)

        suggestion = convert_to_suggestion(traced_name, value, trial, config)

        is_dict = isinstance(suggestion, dict)
        is_list = isinstance(suggestion, list)

        # Handle coupled-sweep-mapping separately
        if is_dict and suggestion.get("type") == "coupled-sweep-mapping":
            coupled_sweep_mapping[traced_name] = suggestion.get("mapping")
            suggestions[param] = {
                "type": suggestion.get("type"),
                "target": suggestion.get("target"),  # for resolving later
            }

        # Handle nested structures that require further processing
        elif (is_dict and suggestion.get("type") != "coupled-sweep-mapping") or (
            is_list and not is_flat_list(suggestion)
        ):
            suggestions[param], coupled_sweep_mapping[traced_name] = get_suggestion(
                config, suggestion, trial, traced_param, coupled_sweep_mapping
            )

        # Handle simple (base) suggestions
        else:
            suggestions[param] = suggestion

        traced_param.pop()

    return suggestions, coupled_sweep_mapping


def resolve_references(
    config: Dict[str, Any],
    node: Dict[str, Any],
    walked_path: List[str],
    coupled_sweep_mapping: Dict[str, Any],
) -> None:
    """Recursively resolve references in a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary. Vaues in this dictionary will be updated through recursion.
        node (Dict[str, Any]): The current node to process.
        walked_path (List[str]): The path that has been walked so far.
        coupled_sweep_mapping (Dict[str, Any]): The mapping values for coupled sweeps.
    """

    def set_value_at_path(
        root: Dict[str, Any], path: List[str], key: Any, value: Any
    ) -> None:
        """Navigate to the parent specified by `path` and set `key` to `value`."""
        parent = root
        for step in path:
            parent = parent[step]
        parent[key] = value

    def resolve_reference(ref_dict: Dict[str, Any]) -> Any:
        """Resolve a 'reference' node."""
        ref_path = ref_dict.get("target")
        return get_value_of_ref(config, ref_path)

    def resolve_coupled_mapping(
        mapping_dict: Dict[str, Any], full_path: List[str]
    ) -> Any:
        """Resolve a 'coupled-sweep-mapping' node."""
        ref_path = mapping_dict.get("target")
        target_value = get_value_of_ref(config, ref_path)

        coupled_node = ".".join(full_path)
        coupled_values = coupled_sweep_mapping.get(coupled_node)
        if coupled_values is None:
            raise ValueError(f"No coupled sweep mapping found for {coupled_node}.")
        if target_value not in coupled_values:
            raise ValueError(
                f"Target value {target_value} not found in coupled sweep mapping for {coupled_node}."
            )
        return coupled_values[target_value]

    # determine items to iterate over
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return

    for key, value in items:
        # recursive cases
        if isinstance(value, list) or (
            isinstance(value, dict)
            and value.get("type") not in ["reference", "coupled-sweep-mapping"]
        ):
            resolve_references(
                config, value, walked_path + [key], coupled_sweep_mapping
            )
            continue

        # reference cases
        if isinstance(value, dict) and value.get("type") == "reference":
            resolved_value = resolve_reference(value)
            set_value_at_path(config, walked_path, key, resolved_value)

        elif isinstance(value, dict) and value.get("type") == "coupled-sweep-mapping":
            resolved_value = resolve_coupled_mapping(value, walked_path + [key])
            set_value_at_path(config, walked_path, key, resolved_value)


def load_yaml(file: Path | str, description: str) -> Dict[str, Any]:
    """Load a YAML file, raising an error if it doesn't exist.

    Args:
        file (Path | str): Path to the YAML file.
        description (str): Description of the file for error messages.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    if not file or not Path(str(file)).exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    custom_loader = QG.config_utils.get_loader()
    with open(file, "r") as f:
        return yaml.safe_load(f, Loader=custom_loader)


def build_search_space_with_dependencies(
    search_space_file: Path,
    depmap_file: Path,
    trial: optuna.trial.Trial,
    tune_model: bool = False,
    tune_training: bool = True,
    base_settings_file: Path = None,
    built_search_space_file: Path = None,
) -> Dict[str, Any]:
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
        Dict[str, Any]: A dictionary representing the hyperparameter search space
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


def get_tuning_settings(tuning_config_path: Path) -> Dict[str, Any]:
    """Get hyperparameter tuning settings from a YAML configuration file.

    Args:
        tuning_config_path (Path): Path to the tuning configuration YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the tuning settings.
    """
    return load_yaml(tuning_config_path, description="Tuning config")


def create_study(tuning_config: Dict[str, Any]) -> None:
    """Create an Optuna study and save to a specified storage.

    Args:
        tuning_config (Dict[str, Any]): The configuration dictionary for tuning settings.
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
