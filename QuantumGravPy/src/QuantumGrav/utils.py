import importlib
from typing import Sequence, Any


def import_and_get(importpath: str) -> Any:
    """Import a module and get an object from it.

    Args:
        importpath (str): The import path of the object to get.

    Returns:
        Any: The name as imported from the module.

    Raises:
        KeyError: When the module indicated by the path is not found
        KeyError: When the object name indidcated by the path is not found in the module
    """
    parts = importpath.split(".")
    module_name = ".".join(parts[:-1])
    object_name = parts[-1]

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise KeyError(f"Importing module {module_name} unsuccessful") from e
    try:
        return getattr(module, object_name)
    except Exception as e:
        raise KeyError(f"Could not load name {object_name} from {module_name}") from e


def assign_at_path(cfg: dict, path: Sequence[Any], value: Any) -> None:
    """Assign a value to a key in a nested dictionary 'dict'. The path to follow through this nested structure is given by 'path'.

    Args:
        cfg (dict): The configuration dictionary to modify.
        path (Sequence[Any]): The path to the key to modify as a list of nodes to traverse.
        value (Any): The value to assign to the key.
    """
    for p in path[:-1]:
        cfg = cfg[p]
    cfg[path[-1]] = value


def get_at_path(cfg: dict, path: Sequence[Any], default: Any = None) -> Any:
    """Get the value at a key in a nested dictionary. The path to follow through this nested structure is given by 'path'.

    Args:
        cfg (dict): The configuration dictionary to modify.
        path (Sequence[Any]): The path to the key to get as a list of nodes to traverse.

    Returns:
        Any: The value at the specified key, or None if not found.
    """
    for p in path[:-1]:
        cfg = cfg[p]

    return cfg.get(path[-1], default)
