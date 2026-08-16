from typing import Any
from collections.abc import Callable
from .base import Callback, Trainable


class Metric(Callback):
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func  # TODO: build a load_type machinery here.

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)

    def before_val(self, trainer: Trainable): ...

    def after_val(self, trainer: Trainable): ...

    def before_test(self, trainer: Trainable): ...

    def after_test(self, trainer: Trainable): ...
