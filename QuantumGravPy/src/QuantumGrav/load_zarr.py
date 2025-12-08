import zarr
from zarr.storage import LocalStore
from pathlib import Path
from typing import Any


def zarr_group_to_dict(group: zarr.Group, target: dict[str, Any]) -> None:
    """Read a nested zarr group into a nested dict.

    Args:
        group (zarr.Group): The zarr group to read.
        target (dict[str, Any]): The target dictionary to populate.

    Returns:
        None
    """
    # go over stuff in the group
    for key in group.keys():
        if isinstance(group[key], zarr.Group):
            if key not in target:
                target[key] = {}
            tgt = target[key]
            zarr_group_to_dict(group[key], tgt)
        else:
            target[key] = group[key][:]


def zarr_file_to_dict(path: Path | str) -> dict[str, Any]:
    """Read a nested zarr group into a nested dict.

    Args:
        path (Path | str): The path to the zarr store.

    Returns:
        dict[str, Any]: A nested dictionary representation of the zarr group.
    """
    # open zarr store and root group
    store = LocalStore(path, read_only=True)
    root = zarr.open_group(store=store, mode="r")

    # target dict
    target: dict[str, Any] = {}

    zarr_group_to_dict(root, target)

    return target
