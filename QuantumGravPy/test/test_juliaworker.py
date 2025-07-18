import QuantumGrav as QG
import pytest
from pathlib import Path


def test_juliaworker_works(ontheflyconfig, jl_vars):
    jlworker = QG.JuliaWorker(
        config=ontheflyconfig,
        **jl_vars,
    )

    assert jlworker.jl_code_path == str(
        Path("./QuantumGravPy/test/julia_testmodule.jl").resolve().absolute()
    )
    assert jlworker.jl_func_name == "Generator"
    assert jlworker.jl_base_module_path == str(
        Path("./QuantumGrav.jl").resolve().absolute()
    )


def test_juliaworker_no_funcname(ontheflyconfig):
    with pytest.raises(ValueError, match="Julia function name must be provided."):
        QG.JuliaWorker(
            config=ontheflyconfig,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_func_name=None,
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_no_codepath(ontheflyconfig):
    with pytest.raises(ValueError, match="Julia code path must be provided."):
        QG.JuliaWorker(
            config=ontheflyconfig,
            jl_code_path=None,
            jl_func_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_jl_failure(ontheflyconfig, mocker):
    # mock the jl_call stuff such that it raises an error
    mock_module = mocker.MagicMock()
    mock_module.seval.side_effect = RuntimeError("Julia call failed.")

    # Mock the newmodule function to return our mock module
    mocker.patch("juliacall.newmodule", return_value=mock_module)

    with pytest.raises(RuntimeError, match="Julia call failed."):
        QG.JuliaWorker(
            config=ontheflyconfig,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_func_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_jl_call(ontheflyconfig):
    jlworker = QG.JuliaWorker(
        config=ontheflyconfig,
        jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
        jl_func_name="Generator",
        jl_base_module_path="./QuantumGrav.jl",
        jl_dependencies=[
            "Distributions",
            "Random",
        ],
    )

    batch = jlworker(5)
    assert len(batch) == 5
    assert all(
        key in batch[0]
        for key in [
            "manifold",
            "boundary",
            "dimension",
            "atomcount",
            "adjacency_matrix",
            "link_matrix",
        ]
    )
