import QuantumGrav as QG
import pytest
from torch_geometric.data import Data


def test_onthefly_dataset_creation_works(
    ontheflyconfig, basic_transform, basic_converter
):
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
        converter=basic_converter,
    )

    assert ontheflydataset.worker is not None
    assert ontheflydataset.transform == basic_transform

    data = ontheflydataset.worker(5)

    assert len(data) == 5
    assert all(
        key in data[0]
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
            converter=lambda x: x,
        )


def test_onthefly_dataset_no_converter(ontheflyconfig):
    with pytest.raises(
        ValueError,
        match="Converter function must be provided to convert Julia objects into standard Python objects.",
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
            transform=lambda x: x,
            converter=None,
        )


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_onthefly_dataset_processing(
    ontheflyconfig, basic_transform, basic_converter, n
):
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
        converter=basic_converter,
    )

    datapoint = ontheflydataset.get(0)

    assert isinstance(datapoint, Data)
    assert datapoint.x.shape[1] == 2  # 2 degrees of freedom
    assert datapoint.y.shape == (3,)  # manifold, boundary, dimension
