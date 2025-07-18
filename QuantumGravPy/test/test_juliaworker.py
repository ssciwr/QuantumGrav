import QuantumGrav as QG
import pytest


def test_juliaworker_works(jlcall_args, jl_vars):
    jlworker = QG.JuliaWorker(
        jl_kwargs=jlcall_args,
        **jl_vars,
    )
    assert jlworker.jl_constructor_name == "Generator"
    assert jlworker.jl_generator is not None


def test_juliaworker_no_funcname(jlcall_args):
    with pytest.raises(ValueError, match="Julia function name must be provided."):
        QG.JuliaWorker(
            jl_kwargs=jlcall_args,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_constructor_name=None,
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_no_codepath(jlcall_args):
    with pytest.raises(ValueError, match="Julia code path must be provided."):
        QG.JuliaWorker(
            jl_kwargs=jlcall_args,
            jl_code_path=None,
            jl_constructor_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_jl_failure(jlcall_args, mocker):
    # mock the jl_call stuff such that it raises an error
    mock_module = mocker.MagicMock()
    mock_module.seval.side_effect = RuntimeError("Julia call failed.")

    # Mock the newmodule function to return our mock module
    mocker.patch("juliacall.newmodule", return_value=mock_module)

    with pytest.raises(RuntimeError, match="Julia call failed."):
        QG.JuliaWorker(
            jl_kwargs=jlcall_args,
            jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
            jl_constructor_name="Generator",
            jl_base_module_path="./QuantumGrav.jl",
            jl_dependencies=[
                "Distributions",
                "Random",
            ],
        )


def test_juliaworker_jl_call(jlcall_args):
    jlworker = QG.JuliaWorker(
        jl_kwargs=jlcall_args,
        jl_code_path="./QuantumGravPy/test/julia_testmodule.jl",
        jl_constructor_name="Generator",
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
