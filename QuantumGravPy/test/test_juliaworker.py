import QuantumGrav as QG
import pytest
from pathlib import Path
from multiprocessing import Pipe, Process
import dill


def test_juliaworker_works(ontheflyconfig):
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


def test_juliaworker_workerloop(ontheflyconfig):
    parent, child = Pipe(duplex=True)
    process = Process(
        target=QG.julia_worker.worker_loop,
        args=[
            child,
            ontheflyconfig,
            "./QuantumGravPy/test/julia_testmodule.jl",
            "Generator",
            "./QuantumGrav.jl",
            [
                "Distributions",
                "Random",
            ],
        ],
    )

    process.start()

    parent.send("GET")

    raw_bytes = parent.recv()
    raw_data = dill.loads(raw_bytes)

    assert len(raw_data) == 5

    parent.send("STOP")
    process.join()
