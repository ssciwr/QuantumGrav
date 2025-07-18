import QuantumGrav as QG
import pytest
from torch_geometric.data import Data


def test_onthefly_dataset_creation_works(
    jlcall_args, jl_vars, basic_transform, basic_converter
):
    ontheflydataset = QG.QGDatasetOnthefly(
        config=jlcall_args,
        transform=basic_transform,
        converter=basic_converter,
        **jl_vars,
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
    assert ontheflydataset.config == jlcall_args
    assert ontheflydataset.worker is not None


def test_onthefly_dataset_no_transform(jlcall_args, jl_vars):
    with pytest.raises(
        ValueError,
        match="Transform function must be provided to turn raw data dictionaries into PyTorch Geometric Data objects.",
    ):
        QG.QGDatasetOnthefly(
            config=jlcall_args,
            jl_code_path=jl_vars["jl_code_path"],
            jl_constructor_name="Generator",
            jl_base_module_path=jl_vars["jl_base_module_path"],
            jl_dependencies=jl_vars["jl_dependencies"],
            transform=None,
            converter=lambda x: x,
        )


def test_onthefly_dataset_no_converter(jlcall_args, jl_vars):
    with pytest.raises(
        ValueError,
        match="Converter function must be provided to convert Julia objects into standard Python objects.",
    ):
        QG.QGDatasetOnthefly(
            config=jlcall_args,
            jl_code_path=jl_vars["jl_code_path"],
            jl_constructor_name="Generator",
            jl_base_module_path=jl_vars["jl_base_module_path"],
            jl_dependencies=jl_vars["jl_dependencies"],
            transform=lambda x: x,
            converter=None,
        )


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_onthefly_dataset_processing(
    jlcall_args, basic_transform, basic_converter, jl_vars, n
):
    # check that the get function works and returns a viable data object
    jlcall_args["n_processes"] = n

    ontheflydataset = QG.QGDatasetOnthefly(
        config=jlcall_args,
        jl_code_path=jl_vars["jl_code_path"],
        jl_constructor_name="Generator",
        jl_base_module_path=jl_vars["jl_base_module_path"],
        jl_dependencies=jl_vars["jl_dependencies"],
        transform=basic_transform,
        converter=basic_converter,
    )

    datapoint = ontheflydataset.get(0)

    assert isinstance(datapoint, Data)
    assert datapoint.x.shape[1] == 2  # 2 degrees of freedom
    assert datapoint.y.shape == (3,)  # manifold, boundary, dimension
