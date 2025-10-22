from abc import abstractmethod, ABC
from typing import Any


class Configurable(ABC):
    @classmethod
    @abstractmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        pass  # must be implemented in subclass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "Configurable":
        pass  # must be implemented in subclass
