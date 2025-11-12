from abc import abstractmethod, ABC
from typing import Any, Dict


class Configurable(ABC):
    """Abstract base class that defines a class to be working with configuration files

    Args:
        ABC (_type_): _description_
    """

    @classmethod
    @abstractmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        """Abstract function that verifies a config file

        Args:
            config (dict[str, Any]): config to verify

        Returns:
            bool: whether the config is valid or not
        """
        pass  # must be implemented in subclass

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
            config (dict[str, Any]): config to intantiate the caller class from

        Returns:
            Configurable: A new instance of the class
        """
        pass  # must be implemented in subclass
