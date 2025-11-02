import pytest 
import zarr 
import numpy as np
import QuantumGrav as QG
import shutil

@pytest.fixture()
def make_test_zarr(tmp_path):   
    # Create a test zarr store with nested groups and arrays
    zarr_path = tmp_path / "test.zarr"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
        
    store = zarr.storage.LocalStore(zarr_path, read_only=False)
    root = zarr.open_group(store=store, mode='w')

    # Create nested groups and arrays
    grp1 = root.create_group("group1",  overwrite=True)
    grp1.create_array("array1", data=np.array([1, 2, 3]))
    grp1.create_array("array2", data=np.array([[1, 2], [3, 4]]))

    grp2 = root.create_group("group2",  overwrite=True)
    subgrp = grp2.create_group("subgroup",  overwrite=True)
    subgrp.create_array("array3", data=np.array([5, 6, 7, 8]))

    yield zarr_path
    
    # Teardown: cleanup after test
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    return tmp_path / "test.zarr"


def test_load_zarr_dict(make_test_zarr): 
    """Test loading a zarr store into a nested dict."""
    zarr_dict =  QG.zarr_to_dict(make_test_zarr)
    dict_to_check = {
        "group1": {
            "array1": np.array([1, 2, 3]),
            "array2": np.array([[1, 2], [3, 4]]),
        },
        "group2": {
            "subgroup": {
                "array3": np.array([5, 6, 7, 8])
            }
        }
    }

    assert "group1" in zarr_dict
    assert "group2" in zarr_dict
    assert "array1" in zarr_dict["group1"]
    assert "array2" in zarr_dict["group1"]
    assert "subgroup" in zarr_dict["group2"]
    assert "array3" in zarr_dict["group2"]["subgroup"]
    assert np.array_equal(zarr_dict["group1"]["array1"], dict_to_check["group1"]["array1"])
    assert np.array_equal(zarr_dict["group1"]["array2"], dict_to_check["group1"]["array2"])
    assert np.array_equal(zarr_dict["group2"]["subgroup"]["array3"], dict_to_check["group2"]["subgroup"]["array3"])

def test_load_zarr_empty(tmp_path): 
    """Test loading an empty zarr store."""
    store = zarr.storage.LocalStore(tmp_path / "empty.zarr")
    zarr.group(store=store, overwrite=True)
    
    zarr_dict = QG.zarr_to_dict(tmp_path / "empty.zarr")
    assert zarr_dict == {}


def test_load_only_arrays(tmp_path): 
    """Test loading a zarr store with only arrays."""
    store = zarr.storage.LocalStore(tmp_path / "only_arrays.zarr")
    root = zarr.group(store=store, overwrite=True)

    root.create_array("array1", data=np.array([1, 2, 3]))
    root.create_array("array2", data=np.array([[1, 2], [3, 4]]))

    zarr_dict = QG.zarr_to_dict(tmp_path / "only_arrays.zarr")
    np.testing.assert_array_equal(zarr_dict["array1"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(zarr_dict["array2"], np.array([[1, 2], [3, 4]]))


def test_load_nd_arrays(tmp_path):
    """Test loading a zarr store with n-dimensional arrays."""
    store = zarr.storage.LocalStore(tmp_path / "nd_arrays.zarr")
    root = zarr.group(store=store, overwrite=True)

    root.create_array("array1", data=np.random.rand(3, 4, 9))
    root.create_array("array2", data=np.random.rand(2, 3, 4, 3, 6))

    zarr_dict = QG.zarr_to_dict(tmp_path / "nd_arrays.zarr")
    assert "array1" in zarr_dict
    assert "array2" in zarr_dict