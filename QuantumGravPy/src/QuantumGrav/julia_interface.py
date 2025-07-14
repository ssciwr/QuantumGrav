import juliacall as jcall
from pathlib import Path
import os
import logging


class JuliaInterface:
    package_path: Path | str | None = None
    module = None
    logger = logging.getLogger(__name__)
    custom_modules: dict[str, Path] | None = None

    def __init__(
        self, package_path: str | Path | None = None, log_level: int = logging.INFO
    ):
        # set up package name
        if package_path is None:
            # README: this seems brittle -> how to do better? -> use installation of quantumgrav locally and use the module path from there?
            self.package_path = (
                Path(os.path.abspath(__file__)).parent.parent.parent / "QuantumGrav.jl"
            )
            self.add_to_load_path(self.package_path)
        else:
            self.package_path = Path(os.path.abspath(package_path))
        self.module = jcall.newmodule("QuantumGravFromPy")
        self._initialize_quantumgrav_module()
        self.logger.setLevel(log_level)
        self.custom_modules = {}

    def add_to_load_path(self, path: str | Path) -> None:
        path = Path(path).resolve()
        self.module.seval(f'push!(LOAD_PATH, "{path}")')
        self.logger.info(f"Added {path} to Julia load path.")

    def load_custom_module(self, path: str | Path) -> None:
        path = Path(path).resolve()
        self.add_to_load_path(path)
        module_name = path.stem
        self.custom_modules[module_name] = jcall.newmodule(module_name)

    def _initialize_quantumgrav_module(
        self,
    ) -> None:
        try:
            self.add_to_load_path(self.package_path)

            # README: this uses the development version of the damn thing, because we don´t have any releases yet
            setup_command = f"""
            using Pkg
            Pkg.develop(path="{self.package_path}/QuantumGrav.jl")
            """
            print(f"evaluating {setup_command}")
            # activate local environment
            self.module.seval(setup_command)

            # load module
            self.module.seval("using QuantumGrav")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize QuantumGrav.jl: {e}") from e

        self.logger.info(
            f"QuantumGrav.jl initialized successfully from {self.package_path}"
        )

    @property
    def quantumgrav_jl_module(self):
        return self.module
