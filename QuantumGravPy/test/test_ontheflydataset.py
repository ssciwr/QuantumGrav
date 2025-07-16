import QuantumGrav as QG
import juliacall as jcall
import pytest
from torch_geometric.data import Data


def test_onthefly_dataset_creation_works(ontheflyconfig, basictransform):
    transform_function = basictransform()

    ontheflydataset = QG.QGDatasetOnthefly(
        config=ontheflyconfig,
        jl_code_path="test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="../../QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
        transform=transform_function,
    )

    assert isinstance(ontheflydataset, jcall.ModuleValue)
    assert isinstance(ontheflydataset.jl_generator, jcall.AnyValue)
    assert ontheflydataset.transform == transform_function

    datapoint = ontheflydataset.jl_generator.__call__(42)

    assert isinstance(datapoint, dict)
    assert all(
        key in datapoint
        for key in [
            "manifold",
            "boundary",
            "dimension",
            "atomcount",
            "adjacency_matrix",
            "link_matrix",
        ]
    )
    assert ontheflydataset.config == ontheflyconfig
    assert ontheflydataset.databatch == []


def test_onthefly_dataset_no_transform(ontheflyconfig):
    with pytest.raises(
        ValueError,
        match="Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects.",
    ):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path="test/julia_testmodule.jl",
            jl_func_name="Generator",
            jl_base_module_path="../../QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=None,
        )


def test_onthefly_dataset_no_funcname(ontheflyconfig):
    with pytest.raises(ValueError, match="Julia function name must be provided."):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path="test/julia_testmodule.jl",
            jl_func_name=None,
            jl_base_module_path="../../QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=lambda x: x,
        )


def test_onthefly_dataset_no_codepath(ontheflyconfig):
    with pytest.raises(ValueError, match="Julia code path must be provided."):
        QG.QGDatasetOnthefly(
            config=ontheflyconfig,
            jl_code_path=None,
            jl_func_name="Generator",
            jl_base_module_path="../../QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
            transform=lambda x: x,
        )


def test_onthefly_dataset_jl_failure(ontheflyconfig, mocker):
    # mock the jl_call stuff such that it raises an error
    with mocker.patch(
        "juliacall.jl_call", side_effect=RuntimeError("Julia call failed.")
    ):
        with pytest.raises(RuntimeError, match="Julia call failed."):
            QG.QGDatasetOnthefly(
                config=ontheflyconfig,
                jl_code_path="test/julia_testmodule.jl",
                jl_func_name="Generator",
                jl_base_module_path="../../QuantumGrav.jl",
                jl_dependencies=[
                    "Distributions",
                    "Random",
                ],
                transform=lambda x: x,
            )


def test_onthefly_dataset_processing_sequential(ontheflyconfig, basic_transform):
    # check that the get function works and returns a viable data object
    ontheflydataset = QG.QGDatasetOnthefly(
        config=ontheflyconfig,
        jl_code_path="test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="../../QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
        transform=basic_transform,
    )
    datapoint = ontheflydataset.get(0)

    assert isinstance(datapoint, Data)
    assert datapoint.x.shape[1] == 4  # 2 degrees + 2 path lengths
    assert datapoint.y.shape == (3,)  # manifold, boundary, dimension


def test_onthefly_dataset_processing_parallel(ontheflyconfig, basic_transform):
    # check that the get function works and returns a viable data object
    ontheflyconfig["n_processes"] = 2  # set to 2 for parallel processing
    ontheflydataset = QG.QGDatasetOnthefly(
        config=ontheflyconfig,
        jl_code_path="test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="../../QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
        transform=basic_transform,
    )
    datapoint = ontheflydataset.get(0)
    assert isinstance(datapoint, Data)
    assert datapoint.x.shape[1] == 4  # 2 degrees + 2 path lengths
    assert datapoint.y.shape == (3,)  # manifold, boundary, dimension
