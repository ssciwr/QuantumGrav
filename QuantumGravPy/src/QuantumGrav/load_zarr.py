import zarr
from zarr.storage import LocalStore
from pathlib import Path
from typing import Any

def _read_zarr(group:zarr.Group, target:dict[str, Any])->None:
    """_summary_

    Args:
        group (zarr.Group): _description_
        target (dict[str, Any]): _description_

    Returns:
        _type_: _description_
    """
    # go over stuff in the group
    for key in group.keys():
        if isinstance(group[key], zarr.Group):
            tgt = target.get(key, dict())
            _read_zarr(group[key], tgt)
        else:
            target[key] = group[key][:]

def zarr_to_dict(path:Path|str)-> dict[str, Any]:
    """_summary_

    Args:
        path (Path | str): _description_

    Returns:
        dict[str, Any]: _description_
    """
    # open zarr store and root group
    store = LocalStore(path, read_only = False)
    root = zarr.create_group(store=store)

    # target dict
    target:dict[str, Any] = {}

    _read_zarr(root, target)

    return target