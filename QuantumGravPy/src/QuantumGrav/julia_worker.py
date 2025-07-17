import juliacall as jcall
import pickle
from multiprocessing import Pipe
from pathlib import Path
from typing import Any


class JuliaWorker:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        jl_code_path: str | Path | None = None,
        jl_func_name: str | None = None,
        jl_module_name: str | None = None,
        jl_base_module_path: str | Path | None = None,
        jl_dependencies: list[str] | None = None,
    ):
        if jl_func_name is None:
            raise ValueError("Julia function name must be provided.")

        if jl_code_path is None:
            raise ValueError("Julia code path must be provided.")

        jl_code_path = Path(jl_code_path).resolve().absolute()
        if not jl_code_path.exists():
            raise FileNotFoundError(f"Julia code path {jl_code_path} does not exist.")

        if jl_module_name is None:
            jl_module_name = jl_code_path.stem

        self.jl_code_path = jl_code_path
        self.jl_func_name = jl_func_name
        self.jl_module_name = jl_module_name
        self.jl_base_module_path = jl_base_module_path
        self.jl_dependencies = jl_dependencies

        try:
            self.jl_module = jcall.newmodule(jl_module_name)
        except Exception as e:
            raise RuntimeError(
                f"Error creating Julia module {jl_module_name}: {e}"
            ) from e

        # add base module for dependencies
        if jl_base_module_path is None:
            raise NotImplementedError("Base module path must be provided at the moment")
        else:
            try:
                self.jl_module.seval(
                    f'using Pkg; Pkg.develop(path="{jl_base_module_path}")'
                )  # only for now -> get from package index later

            except Exception as e:
                raise RuntimeError(
                    f"Error loading base module {jl_base_module_path}: {e}"
                ) from e

        try:
            # add dependencies if provided\
            if jl_dependencies is not None:
                for dep in jl_dependencies:
                    self.jl_module.seval(f'using Pkg; Pkg.add("{dep}")')
        except Exception as e:
            raise RuntimeError(f"Error loading Julia dependencies: {e}") from e

        try:
            # load the julia data generation julia code
            self.jl_module.seval(f'push!(LOAD_PATH, "{jl_code_path}")')
            self.jl_module.seval("using QuantumGrav")
            self.jl_module.seval(f'include("{jl_code_path}")')

            generator_constructor = getattr(self.jl_module, jl_func_name)

            self.jl_generator = generator_constructor(config)
        except Exception as e:
            raise RuntimeError(
                f"Error loading Julia module {jl_module_name}: {e}"
            ) from e

    def __call__(self):
        if self.jl_module is None:
            raise RuntimeError("Julia module is not initialized.")
        raw_data = self.jl_generator()
        return raw_data


def worker_loop(
    pipe: Pipe,
    config: dict[str, Any] | None = None,
    jl_code_path: str | Path | None = None,
    jl_func_name: str | None = None,
    jl_module_name: str | None = None,
    jl_base_module_path: str | Path | None = None,
    jl_dependencies: list[str] | None = None,
):
    print("Worker started", flush=True)
    worker = JuliaWorker(
        config=config,
        jl_code_path=jl_code_path,
        jl_func_name=jl_func_name,
        jl_module_name=jl_module_name,
        jl_base_module_path=jl_base_module_path,
        jl_dependencies=jl_dependencies,
    )

    while True:
        msg = pipe.recv()
        if msg == "STOP":
            break
        elif msg == "GET":
            try:
                result = worker()
                pipe.send(pickle.dumps(result))
                print(f"Worker sent result: {result}", flush=True)
            except Exception as e:
                pipe.send(pickle.dumps(e))
