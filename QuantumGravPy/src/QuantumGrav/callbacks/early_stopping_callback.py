from ..base import Callback, Trainable


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int, monitored_metrics: list[str]):
        self.patience = patience
        self.monitored_metrics = monitored_metrics

    def after_val(self, trainer: Trainable): ...
