from pathlib import Path
from typing import Any
import juliacall as jcall


class JuliaWorker:
    """This class runs a given Julia callable object from a given Julia code file. It additionally imports the QuantumGrav julia module and installs given dependencies if provided. After creation, the wrapped julia callable can be called via the __call__ method of this calls.
    **Warning**: This class requires the juliacall package to be installed in the Python environment.
    **Warning**: This class is in early development and may change in the future, be slow, or otherwis not ready for high performance production use.
    """

    jl_constructor_name = None

    def __init__(
        self,
        jl_kwargs: dict[str, Any] | None = None,
        jl_code_path: str | Path | None = None,
        jl_constructor_name: str | None = None,
        jl_base_module_path: str | Path | None = None,
        jl_dependencies: list[str] | None = None,
    ):
        """Initializes the JuliaWorker with the given parameters.

        Args:
            jl_kwargs (dict[str, Any] | None, optional): Keyword arguments to pass to the Julia callable object constructor. Defaults to None.
            jl_code_path (str | Path | None, optional): Path to the Julia code file in which the callable object is defined. Defaults to None.
            jl_constructor_name (str | None, optional): Name of the Julia constructor function. Defaults to None.
            jl_base_module_path (str | Path | None, optional): Path to the base Julia module 'QuantumGrav.jl'. If not given, tries to load it via a default `using QuantumGrav` import. Defaults to None.
            jl_dependencies (list[str] | None, optional): List of Julia package dependencies. Defaults to None. Will be installed via `Pkg.add` if provided upon first call.

        Raises:
            ValueError: If the Julia function name is not provided.
            ValueError: If the Julia code path is not provided.
            FileNotFoundError: If the Julia code path does not exist.
            NotImplementedError: If the base module path is not provided.
            RuntimeError: If there is an error loading the base module.
            RuntimeError: If there is an error loading Julia dependencies.
            RuntimeError: If there is an error loading the Julia code.
        """

        # we test for a bunch of needed args first
        if jl_constructor_name is None:
            raise ValueError("Julia function name must be provided.")

        if jl_code_path is None:
            raise ValueError("Julia code path must be provided.")

        jl_code_path = Path(jl_code_path).resolve().absolute()
        if not jl_code_path.exists():
            raise FileNotFoundError(f"Julia code path {jl_code_path} does not exist.")

        self.jl_constructor_name = jl_constructor_name
        self.jl_module_name = "QuantumGravPy2Jl"  # the module name is hardcoded here

        # try to initialize the new Julia module,  the do every julia call through thisi module
        try:
            self.jl_module = jcall.newmodule("QuantumGravPy2Jl")

        except jcall.JuliaError as e:
            raise RuntimeError(f"Error creating new julia module: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected exception while creating Julia module {self.jl_module_name}: {e}"
            ) from e

        # add base module for dependencies if exists
        if jl_base_module_path is not None:
            try:
                self.jl_module.seval(
                    f'using Pkg; Pkg.develop(path="{jl_base_module_path}")'
                )  # only for now -> get from package index later
            except jcall.JuliaError as e:
                raise RuntimeError(
                    f"Error loading base module {jl_base_module_path}: {e}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected exception while initializing julia base module: {e}"
                ) from e

        try:
            # add dependencies if provided
            if jl_dependencies is not None:
                for dep in jl_dependencies:
                    self.jl_module.seval(f'using Pkg; Pkg.add("{dep}")')
        except jcall.JuliaError as e:
            raise RuntimeError(f"Error processing Julia dependencies: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected exception while processing Julia dependencies: {e}"
            ) from e

        try:
            # load the julia data generation julia code
            self.jl_module.seval(f'push!(LOAD_PATH, "{jl_code_path}")')
            self.jl_module.seval("using QuantumGrav")
            self.jl_module.seval(f'include("{jl_code_path}")')
            constructor_name = getattr(self.jl_module, jl_constructor_name)
            self.jl_generator = constructor_name(jl_kwargs)
        except jcall.JuliaError as e:
            raise RuntimeError(
                f"Error evaluating Julia code to activate base module: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected exception while loading Julia base module: {e}"
            ) from e

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the wrapped Julia generator with the given arguments.

        Raises:
            RuntimeError: If the Julia module is not initialized.
        Args:
            *args: Positional arguments to pass to the Julia generator.
            **kwargs: Keyword arguments to pass to the Julia generator.
        Returns:
            Any: The raw data generated by the Julia generator.
        """
        if self.jl_module is None:
            raise RuntimeError("Julia module is not initialized.")
        raw_data = self.jl_generator(*args, **kwargs)
        return raw_data
