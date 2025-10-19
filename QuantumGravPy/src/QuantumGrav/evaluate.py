import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
import logging
from abc import abstractmethod
from sklearn.metrics import f1_score


class DefaultEvaluator:
    """Default evaluator for model evaluation - testing and validation during training"""

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        compute_per_task: dict[int, dict[Any, Callable]],
        get_target_per_task: dict[
            int, Callable[[dict[int, torch.tensor], int], torch.tensor]
        ]
        | None = None,
        apply_model: Callable | None = None,
    ):
        """Default evaluator for model evaluation.

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable): A function to apply the model to the data.
        """
        self.criterion = criterion
        self.apply_model = apply_model
        self.device = device
        self.data: pd.DataFrame | None = None
        self.logger = logging.getLogger(__name__)
        self.compute_per_task = compute_per_task
        self.get_target_per_task = get_target_per_task
        self.active_tasks = []

    def reset(self) -> None:
        """Reset the evaluator's recorded data."""
        self.data = None
        self.active_tasks = []

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> None:
        """Evaluate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to evaluate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for evaluation.

        Returns:
             list[Any]: A list of evaluation results.
        """
        model.eval()
        current_data = dict()

        self.active_tasks = []
        for i, task in enumerate(model.active_tasks):
            if task:
                self.active_tasks.append(i)

        # apply model on validation data
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data = batch.to(self.device)
                if self.apply_model:
                    outputs = self.apply_model(model, data)
                else:
                    outputs = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(outputs, data)
                current_data["loss"] = current_data.get("loss", []) + [loss.item()]

                # record outputs and targets per task
                for i, out in outputs.items():
                    y = self.get_target_per_task[i](data, i)
                    current_data[f"output_{i}"] = current_data.get(
                        f"output_{i}", []
                    ) + [out.cpu()]  # append outputs
                    current_data[f"target_{i}"] = current_data.get(
                        f"target_{i}", []
                    ) + [y.cpu()]  # append targets

        # compute metrics
        for i, task in self.compute_per_task.items():
            if i in self.active_tasks:
                for name, func in task.items():
                    if name not in current_data:
                        current_data[name] = func(current_data, i)

        # add current_data to self.data
        if self.data is None:
            self.data = pd.DataFrame(current_data)
        else:
            # when the dataframe has not all columns present yet, add them with NaN values
            # this can happen when the active tasks change during training
            for k in current_data.keys():
                if k not in self.data.columns:
                    self.data[k] = np.nan

            # add the necessary data
            self.data = pd.concat(
                [self.data, pd.DataFrame(current_data)], ignore_index=True
            )

    @abstractmethod
    def report(self) -> None:
        """Report the evaluation results.

        Args:
            data (pd.DataFrame | torch.Tensor | list | dict): Evaluation results to report.
        """
        pass

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefaultEvaluator":
        """Create DefaultEvaluator from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'device', 'criterion', 'compute_per_task', 'get_target_per_task', and 'apply_model'.

        Returns:
            DefaultEvaluator: An instance of DefaultEvaluator initialized with the provided configuration.
        """
        # TODO
        pass


class DefaultTester(DefaultEvaluator):
    """Default tester for model testing.

    Args:
        DefaultEvaluator (Class): Inherits from DefaultEvaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def test(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> None:
        """Test the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to test.
            data_loader (torch_geometric.loader.DataLoader): Data loader for testing.
        """
        self.evaluate(model, data_loader)

    def report(self) -> None:
        """Report the testing results."""
        if self.data is not None:
            self.logger.info("Testing Results:")
            self.logger.info(self.data.tail(1).to_string(index=False))
        else:
            self.logger.info("No testing data to report.")


class DefaultValidator(DefaultTester):
    """Default validator for model validation.

    Args:
        DefaultEvaluator (Class): Inherits from DefaultEvaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def validate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> None:
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        """
        self.evaluate(model, data_loader)

    def report(self) -> None:
        """Report the validation results."""
        if self.data is not None:
            self.logger.info("Validation Results:")
            self.logger.info(self.data.tail(1).to_string(index=False))
        else:
            self.logger.info("No validation data to report.")


class F1ScoreEval:
    """F1Score evaluator, useful for evaluation of classification problems. A callable class that builds an f1 evaluator to a given set of specifications. Uses sklearn.metrics.f1_score for the computation fo the f1 score."""

    def __init__(
        self,
        average: str = "macro",
        labels: list[int] | None = None,
    ):
        """F1 score evaluator.

        Args:
            average (str, optional): Averaging method for F1 score. Defaults to "macro". Can be 'micro', 'macro', 'weighted', or 'none'.
        """
        self.average = average
        self.labels = labels

    def __call__(
        self, data: dict[Any, Any], task: int
    ) -> float | np.float64 | np.array | list:
        """Compute F1 score for a given task.

        Args:
            data (dict[Any, Any]): Dictionary containing outputs and targets.
            task (int): Task index.
        pass
        """
        # TODO: check that the dimensionality is still correct
        # if the model output has more than one dimension, we need to adjust the y_pred accordingly
        # and squeeze/unsqueeze as needed
        y_true = torch.cat(data[f"target_{task}"])
        y_pred = torch.cat(data[f"output_{task}"])
        return f1_score(y_true, y_pred, average=self.average, labels=self.labels)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "F1ScoreEval":
        """Create F1ScoreEval from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'average', 'mode', and 'labels'.
        Returns:
            F1ScoreEval: An instance of F1ScoreEval initialized with the provided configuration.
        """
        return cls(
            average=config.get("average", "macro"),
            mode=config.get("mode", "binary"),
            labels=config.get("labels", None),
        )


class AccuracyEval:
    """Accuracy evaluator, primarily useful for evaluation of regression problems. A callable class that builds an accuracy evaluator to a given set of specifications."""

    def __init__(self, metrics: Callable | None = None):
        """Accuracy evaluator initialization."""

        if metrics is None:
            self.metrics = torch.nn.MSELoss()
        else:
            self.metrics = metrics

    def __call__(
        self, data: dict[Any, Any], task: int
    ) -> float | np.float64 | np.array | list:
        """Compute accuracy for a given task.

        Args:
            data (dict[Any, Any]): Dictionary containing outputs and targets.
            task (int): Task index.
        """
        y_true = torch.cat(data[f"target_{task}"])
        y_pred = torch.cat(data[f"output_{task}"])

        return self.metrics(y_pred, y_true).item()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AccuracyEval":
        """Create AccuracyEval from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'average', 'mode', and 'labels'.
        Returns:
            AccuracyEval: An instance of AccuracyEval initialized with the provided configuration.
        """
        # TODO: add metrics lookup
        return cls(
            metrics=None,
        )


# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore
class DefaultEarlyStopping:
    """Early stopping based on a validation metric."""

    # put this into the package
    def __init__(
        self,
        patience: int,
        delta: list[float] = [1e-4],
        window: list[int] = [7],
        metric: list[str] = ["loss"],
        smoothing: bool = False,
        criterion: Callable = lambda early_stopping_instance,
        data: early_stopping_instance.current_patience <= 0,
        init_best_score: float = np.inf,
        mode: str | Callable[[list[bool]], bool] = "any",
        grace_period: list[int] = [
            0,
        ],
    ):
        """Early stopping initialization.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float, optional): Minimum change to consider an improvement. Defaults to 1e-4.
            window (int, optional): Size of the moving window for smoothing. Defaults to 7.
            metric (str, optional): Metric to monitor for early stopping. Defaults to "loss". This class always assumes that lower values for 'metric' are better.
            smoothing (bool, optional): Whether to apply smoothed mean to the metric to dampen fluctuations. Defaults to False.
            criterion (Callable, optional): Custom stopping criterion. Defaults to a function that stops when patience is exhausted.
            init_best_score (float): initial best score value.
            mode (str | Callable[[list[bool]], bool], optional): The mode for early stopping. Can be "any", "all", or a custom function. Defaults to "any". This decides wheather all tracked metrics have to improve or only one of them, or if something else should be done by applying a custom function to the array of evaluation results.
        """
        lw = len(window)
        lm = len(metric)
        ld = len(delta)
        lg = len(grace_period)

        if min([lw, lm, ld, lg]) != max([lw, lm, ld, lg]):
            raise ValueError(
                f"Inconsistent lengths for early stopping parameters: {lw}, {lm}, {ld}, {lg}"
            )

        self.patience = patience
        self.current_patience = patience
        self.delta = delta
        self.window = window
        self.found_better = [False for _ in range(lw)]
        self.metric = metric
        self.init_best_score = init_best_score
        self.best_score = [init_best_score for _ in range(lw)]
        self.smoothing = smoothing
        self.logger = logging.getLogger(__name__)
        self.criterion = criterion
        self.mode = mode
        self.found_better = [False for _ in range(lw)]
        self.grace_period = grace_period
        self.current_grace_period = [grace_period[i] for i in range(lg)]

    @property
    def found_better_model(self) -> bool:
        """Check if a better model has been found."""

        if self.mode == "any":
            return any(self.found_better)
        elif self.mode == "all":
            return all(self.found_better)
        elif callable(self.mode):
            return self.mode(self.found_better)
        else:
            raise ValueError("Mode must be 'any', 'all', or a callable in Evaluator")

    def reset(self) -> None:
        """Reset early stopping state."""
        self.current_patience = self.patience
        self.found_better = [False for _ in range(len(self.window))]
        self.current_grace_period = self.grace_period

    def add_task(
        self, delta: float, window: int, metric: str, grace_period: int
    ) -> None:
        """Add a new task for early stopping.

        Args:
            delta (float): Minimum change to consider an improvement.
            window (int): Size of the moving window for smoothing.
            metric (str): Metric to monitor for early stopping.
            grace_period (int): Grace period for early stopping.
        """
        self.delta.append(delta)
        self.window.append(window)
        self.metric.append(metric)
        self.grace_period.append(grace_period)
        self.current_grace_period.append(grace_period)
        self.best_score.append(self.init_best_score)
        self.found_better.append(False)

    def remove_task(self, index: int) -> None:
        """Remove a task from early stopping.

        Args:
            index (int): The index of the task to remove.
        """
        self.delta.pop(index)
        self.window.pop(index)
        self.metric.pop(index)
        self.grace_period.pop(index)
        self.best_score.pop(index)

    def __call__(self, data: pd.DataFrame | pd.Series) -> bool:
        """Evaluate early stopping criteria. This is done by comparing the last value of data[self.metric] with the current best value recorded. If that value is better than the current best, the current best is updated,
        patience is reset and 'found_better' is set to True. Otherwise, if the number of datapoints in 'data' is greater than self.window, the patience is decremented.

        Args:
            data (pd.DataFrame | pd.Series): Recorded evaluation metrics in a pandas structure.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        self.found_better = [False for _ in range(len(self.window))]
        ds = {}  # dict to hold current metric values

        # go over all registered metrics and check if the model performs better on any of them
        # then aggregrate the result with 'found_better_model'.
        for i in range(len(self.window)):
            # prevent a skipped metric from affecting early stopping
            if self.metric[i] not in data.columns:
                self.logger.warning(f"    Metric {self.metric[i]} not found in data.")
                self.found_better[i] = True
                continue

            if self.smoothing:
                d = (
                    data[self.metric[i]]
                    .rolling(window=self.window[i], min_periods=1)
                    .mean()
                )
            else:
                d = data[self.metric[i]]

            if self.best_score[i] - self.delta[i] > d.iloc[-1] and len(
                data
            ):  # always minimize the metric
                self.logger.info(
                    f"    Better model found at task {i}: {d.iloc[-1]:.8f}, current best: {self.best_score[i]:.8f}"
                )
                self.found_better[i] = True

            ds[i] = d.iloc[-1]

        if self.found_better_model:
            # when we found a better model the stopping patience gets reset
            self.logger.info("Found better model")
            for i in range(len(self.best_score)):
                if self.found_better[i] and i in ds:
                    self.logger.info(
                        f"current best score: {self.best_score[i]:.8f}, current score: {ds[i]:.8f}"
                    )
                    self.best_score[i] = ds[i]  # record best score
            self.current_patience = self.patience  # reset patience

        # only when all grace periods are done will we reduce the patience
        elif all([g <= 0 for g in self.current_grace_period]):
            self.current_patience -= 1
        else:
            pass
            # don't do anything here, we want at least 'grace_period' many epochs before patience is reduced

        for i in range(len(self.window)):
            self.logger.info(
                f"EarlyStopping: current patience: {self.current_patience}, best score: {self.best_score[i]:.8f}, grace_period: {self.current_grace_period[i]}"
            )

        for i in range(len(self.current_grace_period)):
            if self.current_grace_period[i] > 0:
                self.current_grace_period[i] -= 1

        return self.criterion(self, data)
