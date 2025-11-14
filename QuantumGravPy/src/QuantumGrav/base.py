from abc import abstractmethod, ABC
from typing import Any, Collection, Dict, Mapping

import torch


class Configurable(ABC):
    """Abstract base class for objects that can be configured via configuration files.
    Subclasses must implement methods to verify, serialize, and instantiate from configuration dictionaries.
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
            config (dict[str, Any]): config to instantiate the caller class from

        Returns:
            Configurable: A new instance of the class
        """
        pass  # must be implemented in subclass


class BaseModel(torch.nn.Module, Configurable):
    """Abstract base class for QuantumGrav models.
    Subclasses must implement methods to verify, serialize, and instantiate from configuration dictionaries.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor | Mapping[int, torch.Tensor] | Collection[torch.Tensor]:
        """Abstract forward method for the model.

        Returns:
            torch.Tensor: output tensor of the model
        """
        pass  # must be implemented in subclass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model's state to file.

        Args:
            path (str): path to save the model to.
        """

        pass

    @abstractmethod
    @classmethod
    def load(
        cls, path: str, device: torch.device = torch.device("cpu")
    ) -> "BaseModel":
        """Load a mode instance from file

        Args:
            path (str): Path to the file to load.
            device (torch.device): device to put the model to. Defaults to torch.device("cpu")
        Returns:
            BaseModel: A BaseModel instance initialized from the data loaded from the file.
        """
        pass  # must be implemented in subclass
