from abc import abstractmethod, ABC
from typing import Any


class Configurable(ABC):
    """Abstract base class for objects that can be configured via configuration files.
    Subclasses must implement methods to verify, serialize, and instantiate from configuration dictionaries.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Configurable:
        """Instantiate the caller class from a config

        Args:
            config (dict[str, Any]): config to instantiate the caller class from

        Returns:
            Configurable: A new instance of the class
        """
        pass  # must be implemented in subclass


class Trainable(Configurable):
    @classmethod
    def from_config(cls, config: dict[str, Any]):
        """Instantiate the caller class from a config

        Args:
            config (dict[str, Any]): config to instantiate the caller class from

        Returns:
            Configurable: A new instance of the class
        """
        return cls(**config)

    @abstractmethod
    def train(self, train_dataset): ...

    @abstractmethod
    def test(self, train_dataset): ...

    @abstractmethod
    def save_snapshot(self): ...

    @abstractmethod
    @classmethod
    def load_snapshot(cls, path: str, new_path: str): ...


class Callback(Configurable):
    def __init__(
        self,
    ): ...

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)

    def before_epoch(self, trainer: Trainable): ...

    def after_epoch(self, trainer: Trainable): ...

    def before_val(self, trainer: Trainable): ...

    def after_val(self, trainer: Trainable): ...

    def before_test(self, trainer: Trainable): ...

    def after_test(self, trainer: Trainable): ...

    def before_train(self, trainer: Trainable): ...

    def after_train(self, trainer: Trainable): ...
