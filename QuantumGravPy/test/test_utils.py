import QuantumGrav as QG
import pytest
import torch
from torch_geometric.nn import global_mean_pool


def test_assign_at_path():
    testdict = {
        "a": {
            "b": {
                "c": 3,
            }
        },
        "d": 42,
    }

    QG.utils.assign_at_path(
        testdict,
        [
            "a",
            "b",
            "c",
        ],
        12,
    )
    assert testdict["a"]["b"]["c"] == 12

    QG.utils.assign_at_path(
        testdict,
        [
            "a",
            "b",
        ],
        {"x": 42},
    )

    assert testdict["a"]["b"] == {"x": 42}

    with pytest.raises(KeyError):
        QG.utils.assign_at_path(testdict, ["v", "z"], 12)


def test_get_at_path():
    testdict = {
        "a": {
            "b": {
                "c": 3,
            }
        },
        "d": 42,
        "r": {3: "v"},
    }

    assert QG.utils.get_at_path(testdict, ["a", "b", "c"]) == 3

    assert QG.utils.get_at_path(testdict, ["r", 3]) == "v"

    with pytest.raises(KeyError):
        QG.utils.get_at_path(
            testdict,
            ["x", "v"],
        )


def test_import_and_get():
    """Test the import_and_get function."""
    # Test importing a standard library class
    result = QG.utils.import_and_get("torch.nn.Linear")
    assert result is torch.nn.Linear

    # Test importing a function
    result = QG.utils.import_and_get("torch.cat")
    assert result is torch.cat

    # Test importing from torch_geometric
    result = QG.utils.import_and_get("torch_geometric.nn.global_mean_pool")
    assert result is global_mean_pool

    # Test invalid module path
    with pytest.raises(KeyError, match="Importing module .* unsuccessful"):
        QG.utils.import_and_get("nonexistent.module.Class")

    # Test invalid object name
    with pytest.raises(KeyError, match="Could not load name .* from"):
        QG.utils.import_and_get("torch.nn.NonExistentClass")
