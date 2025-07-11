import juliacall as jcall
from pathlib import Path
import os
import logging


class JuliaInterface:
    """Interface for interacting with the Julia programming language. This class provides methods to initialize the QuantumGrav.jl package, add directories to the Julia load path, and execute Julia code from Python."""

    def __init__(
        self, package_path: str | Path | None = None, log_level: int = logging.INFO
    ):
        """Initialize the Julia interface for QuantumGrav.

        Args:
            package_path (str | Path | None, optional): Path to the QuantumGrav.jl package. Defaults to None.
            log_level (int, optional): Logging level. Defaults to logging.INFO.
        """
        # set up package name
        if package_path is None:
            self.package_path = (
                Path(os.path.abspath(__file__)).parent.parent.parent / "QuantumGrav.jl"
            )
        else:
            self.package_path = Path(os.path.abspath(package_path))
        self.module = jcall.newmodule("QuantumGravFromPy")
        self._initialize_quantumgrav_julia()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def add_to_load_path(self, path: str | Path) -> None:
        """
        Add a directory to the Julia load path.
        """
        path = Path(path).resolve()
        jcall.seval(f'push!(LOAD_PATH, "{path}")')
        self.logger.info(f"Added {path} to Julia load path.")

    def load_custom_module(self, path: str | Path) -> None:
        """
        Load a custom Julia module from the specified path.
        """
        path = Path(path).resolve()
        self.add_to_load_path(path)
        module_name = path.stem
        self.module.seval(f"using {module_name}")

    def _initialize_quantumgrav_module(
        self,
    ) -> None:
        """Initialize the Julia interface for QuantumGrav.

        Raises:
            RuntimeError: If initialization fails, e.g., if the package is not found.
        """
        try:
            self.add_to_load_path(self.package_path)

            # activate local environment
            jcall.seval('using Pkg; Pkg.activate("$(self.package_path)")')

            # load module
            self.module.seval("using QuantumGrav")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize QuantumGrav.jl: {e}") from e

        self.logger.info(
            f"QuantumGrav.jl initialized successfully from {self.package_path}"
        )

    @property
    def quantumgrav_jl_module(self):
        """Get the Julia module object for interacting with QuantumGrav.jl"""
        return self.module
