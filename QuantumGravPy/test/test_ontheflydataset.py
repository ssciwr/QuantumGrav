import QuantumGrav as QG
import pytest
from torch_geometric.data import Data
import pickle


def test_onthefly_dataset_creation_works(ontheflyconfig, basic_transform):
    ontheflydataset = QG.QGDatasetOnthefly(
        config=ontheflyconfig,
        jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="./QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
        transform=basic_transform,
    )

    assert ontheflydataset.worker is not None
    assert ontheflydataset.transform == basic_transform

    ontheflydataset.parent_conn.send("GET")
    data = ontheflydataset.parent_conn.recv()
    data = pickle.loads(data)

    # assert len(data) == 5
    # assert all(
    #     key in data[0]
    #     for key in [
    #         "manifold",
    #         "boundary",
    #         "dimension",
    #         "atomcount",
    #         "adjacency_matrix",
    #         "link_matrix",
    #     ]
    # )
    # assert ontheflydataset.config == ontheflyconfig

    ontheflyconfig.shutdown()


def test_onthefly_dataset_no_transform(ontheflyconfig):
    with pytest.raises(
        ValueError,
        match="Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects.",
    ):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_func_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=None,
        )


def test_onthefly_dataset_no_funcname(ontheflyconfig, basic_transform):
    with pytest.raises(ValueError, match="Julia function name must be provided."):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_func_name=None,
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=basic_transform,
        )


def test_onthefly_dataset_no_codepath(ontheflyconfig, basic_transform):
    with pytest.raises(ValueError, match="Julia code path must be provided."):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path=None,
            jl_func_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=basic_transform,
        )


def test_onthefly_dataset_jl_failure(ontheflyconfig, mocker):
    # mock the jl_call stuff such that it raises an error
    mock_module = mocker.MagicMock()
    mock_module.seval.side_effect = RuntimeError("Julia call failed.")

    # Mock the newmodule function to return our mock module
    mocker.patch("juliacall.newmodule", return_value=mock_module)

    with pytest.raises(RuntimeError, match="Julia call failed."):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_func_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=lambda x: x,
        )


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_onthefly_dataset_processing(ontheflyconfig, basic_transform, n):
    # check that the get function works and returns a viable data object
    ontheflyconfig["n_processes"] = n

    ontheflydataset = QG.QGDatasetOnthefly(
        config=ontheflyconfig,
        jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="./QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
        transform=basic_transform,
    )

    datapoint = ontheflydataset.get(0)

    assert isinstance(datapoint, Data)
    assert datapoint.x.shape[1] == 2  # 2 degrees of freedom
    assert datapoint.y.shape == (3,)  # manifold, boundary, dimension
