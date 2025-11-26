from abc import abstractmethod, ABC
from typing import Any, Dict


class Configurable(ABC):
    """Abstract base class for objects that can be configured via configuration files.
    Subclasses must implement methods to verify, serialize, and instantiate from configuration dictionaries.
    """

    @abstractmethod
    def to_config(self) -> Dict[Any, Any]:
        """Convert the caller to a config

        Returns:
            Dict[Any, Any]: config representation of the caller instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "Configurable":
        """Instantiate the caller class from a config

        Args:
            config (dict[str, Any]): config to instantiate the caller class from

        Returns:
            Configurable: A new instance of the class
        """
        pass  # must be implemented in subclass
