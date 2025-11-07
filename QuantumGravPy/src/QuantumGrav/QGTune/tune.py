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
    is_tuple_of_3 = isinstance(value, tuple) and len(value) == 3
    two_floats = (
        (isinstance(value[0], float) and isinstance(value[1], float))
        if is_tuple_of_3
        else False
    )
    third_is_float_or_bool = (
        (isinstance(value[2], float) or isinstance(value[2], bool))
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
    is_tuple_of_3 = isinstance(value, tuple) and len(value) == 3
    all_ints = all(isinstance(v, int) for v in value) if is_tuple_of_3 else False
    return is_tuple_of_3 and all_ints


def get_value_of_ref(config: Dict[str, Any], ref_path: List[Any]) -> Any:
    """Get the value pointed by a reference path in the configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        ref_path (List[Any]): The reference path to a position in the config.
            E.g. `gcn_net[0].out_dim`

    Returns:
        Any: The value pointed by the reference path.
    """
    current = config
    for part in ref_path:
        try:
            current = current[part]
        except Exception:
            raise ValueError(f"Invalid reference path: {ref_path}")
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
    is_sweep = isinstance(node, dict) and node.get("type") == "sweep"
    is_coupled_sweep = isinstance(node, dict) and node.get("type") == "coupled-sweep"
    is_range = isinstance(node, dict) and node.get("type") == "range"

    if is_range:
        node_values = node.get("tune_values")
    elif is_sweep or is_coupled_sweep:
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
        target_node = get_value_of_ref(config, node.get("target"))
        if target_node.get("type") != "sweep":
            raise ValueError(
                f"Target of coupled-sweep {node.get('target')} is not a sweep node."
            )
        target_values = target_node.get("values")
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get a dictionary of suggestions from a configuration dictionary.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        current_node (Dict[str, Any] | List): The current configuration node.
        trial (optuna.trial.Trial): The Optuna trial object.
        traced_param (list, optional): The list of traced parameters.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - A dictionary of suggestions.
            - A dictionary mapping coupled sweep targets to their values.
                Key is the full path to the coupled sweep node.
                Value is the mapping dict between target values and coupled values.
    """
    if isinstance(current_node, list):
        items = enumerate(current_node)
        suggestions = [None] * len(current_node)
    elif isinstance(current_node, dict):
        items = current_node.items()
        suggestions = {}
    else:
        return {}

    # store coupled sweep mappings under the current node
    # key is full path to the coupled sweep node
    # value is the mapping dict between target values and coupled values
    # e.g. {"model.bar.0.x": { -1: -10, -2: -20 }
    coupled_sweep_mapping = {}

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
            suggestions[param], tmp_coupled_mapping = get_suggestion(
                config, suggestion, trial, traced_param
            )
            if tmp_coupled_mapping:
                coupled_sweep_mapping.update(tmp_coupled_mapping)

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
        config (Dict[str, Any]): The configuration dictionary.
            Values in this dictionary will be updated through recursion.
        node (Dict[str, Any]): The current node in the config to process.
        walked_path (List[str]): The path that has been walked so far.
        coupled_sweep_mapping (Dict[str, Any]): The mapping values for coupled sweeps.
    """

    def set_value_at_path(root: Dict[str, Any], path: List[str], value: Any) -> None:
        """Navigate to the parent specified by `path` and set `key` to `value`."""
        parent = root
        for step in path[:-1]:
            parent = parent[step]
        parent[path[-1]] = value

    def resolve_reference(ref_dict: Dict[str, Any]) -> Any:
        """Resolve a 'reference' node."""
        ref_path = ref_dict.get("target")
        return get_value_of_ref(config, ref_path)

    def resolve_coupled_mapping(
        mapping_dict: Dict[str, Any], full_path: List[str | int]
    ) -> Any:
        """Resolve a 'coupled-sweep-mapping' node."""
        ref_path = mapping_dict.get("target")
        target_value = get_value_of_ref(config, ref_path)

        coupled_node = ".".join([str(part) for part in full_path])
        coupled_values = coupled_sweep_mapping.get(coupled_node)
        if coupled_values is None:
            raise ValueError(f"No coupled sweep mapping found for {coupled_node}.")
        if target_value not in coupled_values:
            raise ValueError(
                f"Target value {target_value} not found in coupled sweep mapping for {coupled_node}."
            )
        return coupled_values[target_value]

    # reference cases
    if isinstance(node, dict) and node.get("type") == "reference":
        resolved_value = resolve_reference(node)
        set_value_at_path(config, walked_path, resolved_value)
        return

    elif isinstance(node, dict) and node.get("type") == "coupled-sweep-mapping":
        resolved_value = resolve_coupled_mapping(node, walked_path)
        set_value_at_path(config, walked_path, resolved_value)
        return

    # recursive cases
    # determine items to iterate over
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return

    for key, value in items:
        resolve_references(config, value, walked_path + [key], coupled_sweep_mapping)


def load_yaml(file: Path | str) -> Dict[str, Any]:
    """Load a YAML file, raising an error if it doesn't exist.

    Args:
        file (Path | str): Path to the YAML file.

    Returns:
        Dict[str, Any]: The contents of the YAML file as a dictionary.
    """
    if not file or not Path(str(file)).exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    custom_loader = QG.config_utils.get_loader()
    with open(file, "r") as f:
        return yaml.load(f, Loader=custom_loader)


def build_search_space(config_file: Path, trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Build a hyperparameter search space from a YAML configuration file
    and an Optuna trial object.

    Args:
        config_file (Path): Path to the configuration YAML file.
        trial (optuna.trial.Trial): The Optuna trial object.

    Returns:
        Dict[str, Any]: A dictionary representing the search space
            with resolved references.
    """
    config = load_yaml(config_file)

    search_space, coupled_sweep_mapping = get_suggestion(
        config=config, current_node=config, trial=trial, traced_param=[]
    )

    resolve_references(
        config=search_space,
        node=search_space,
        walked_path=[],
        coupled_sweep_mapping=coupled_sweep_mapping,
    )

    return search_space


def create_study(tuning_config: Dict[str, Any]) -> optuna.study.Study:
    """Create an Optuna study and save to a specified storage.

    Args:
        tuning_config (Dict[str, Any]): The configuration dictionary for tuning settings.

    Returns:
        optuna.study.Study: The created Optuna study.
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


def convert_to_pyobject_tags(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert specific values in the configuration dictionary to `pyobject` YAML tags.
    This is useful for saving the best configuration back to a YAML file,
    and make sure that these tags are converted back to the original structures
    when the YAML file is loaded again.

    `pyobject` tags can only be applied to values that are not of built-in types.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Dict[str, Any]: The configuration dictionary with converted custom tags.
    """
    # determine items to iterate over
    if isinstance(config, dict):
        items = config.items()
        new_config = {}
    elif isinstance(config, list):
        items = enumerate(config)
        new_config = [None] * len(config)
    else:
        return config

    for key, value in items:
        is_dict = isinstance(value, dict)
        is_list = isinstance(value, list)

        # recursive cases
        if is_dict or is_list:
            new_value = convert_to_pyobject_tags(value)
            new_config[key] = new_value

        # base cases
        else:
            # convert only non-built-in types to pyobject tags
            if not isinstance(value, (bool, str, float, int, list, dict)):
                # convert to !pyobject name_of_class
                module = value.__module__
                class_name = value.__name__
                full_class_name = f"{module}.{class_name}"
                pyobject_tag = f"!pyobject {full_class_name}"
                new_config[key] = pyobject_tag
            else:
                new_config[key] = value

    return new_config


def save_best_config(
    config_file: Path,
    best_trial: optuna.trial.FrozenTrial,
    output_file: Path,
):
    """Save the best configuration by combining the best trial parameters
    with the original config file.

    Args:
        config_file (Path): Path to the original configuration YAML file.
        best_trial (optuna.trial.FrozenTrial): The best trial object.
        output_file (Path): Path to the output YAML file for the best configuration.
    """
    if not output_file:
        raise ValueError("Output file path must be provided.")

    # create search space with references and coupled sweeps again from the config file
    config = load_yaml(config_file)
    search_space_w_refs, coupled_sweep_mapping = get_suggestion(
        config=config, current_node=config, trial=best_trial, traced_param=[]
    )
    best_trial_params = best_trial.params

    best_config = copy.deepcopy(search_space_w_refs)

    def get_next(current: dict, part: str | int):
        try:
            index = int(part)
            return current[index]
        except ValueError:
            return current.get(part)

    def set_next(current: dict, part: str | int, value: Any):
        try:
            index = int(part)
            current[index] = value
        except ValueError:
            current[part] = value

    for key, value in best_trial_params.items():
        parts = key.split(".")
        current = best_config
        for part in parts[:-1]:
            current = get_next(current, part)
        set_next(current, parts[-1], value)

    # apply dependencies to resolve any changes
    resolve_references(
        config=best_config,
        node=best_config,
        walked_path=[],
        coupled_sweep_mapping=coupled_sweep_mapping,
    )

    # convert specific values to pyobject tags
    best_config = convert_to_pyobject_tags(best_config)

    with open(output_file, "w") as file:
        yaml.safe_dump(best_config, file)

    print(f"Best configuration saved to {output_file}.")
