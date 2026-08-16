from typing import Any
import torch
from .base import Trainable


class AccTrainer(Trainable):

    def __init__(
        self,
    ): ...

    def _build_callbacks(self, args: dict[str, list[Any]], kwargs: dict[str, dict[str, Any]]):
        ...

    def _prepare_callbacks(self, callbacks):
        ...

    def _apply_callbacks(self, funcname: str):
        for callback in self.callbacks:
            if funcname in callback:
                func = getattr(callback, funcname)
                func(self)

    def _build_model(self):
        ...

    def _build_optimizer(self):
        ...

    def _build_lr_scheduler(self):
        ...

    @classmethod
    def from_config(cls, config: dict[str, Any]): ...

    def _train_epoch(self, train_dataset): ...
        self.model.train()

    @torch.no_grad()
    def validate(self, val_dataset): ...
        self.model.eval()

    def save_snapshot(self): ...

    @classmethod
    def load_snapshot(cls, path: str, new_path: str): ...

    def train(self, train_dataset): ...

    def test(self, test_dataset): ...

