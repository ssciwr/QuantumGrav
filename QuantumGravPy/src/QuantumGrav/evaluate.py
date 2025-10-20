import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
import logging
from abc import abstractmethod
from sklearn.metrics import f1_score
from jsonschema import validate, ValidationError

from . import utils
from . import base


class DefaultEvaluator(base.Configurable):
    """Default evaluator for model evaluation - testing and validation during training"""

    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Evaluator Configuration",
        "type": "object",
        "properties": {
            "device": {
                "type": "string",
                "description": "The device to run the evaluation on.",
            },
            "criterion": {
                "type": "string",
                "description": "The loss function to use for evaluation.",
            },
            "compute_per_task": {
                "type": "object",
                "description": "Task-specific metrics to compute.",
                "additionalProperties": {},
            },
            "get_target_per_task": {
                "type": "object",
                "description": "Function to get the target for each task.",
                "additionalProperties": {},
            },
        },
        "required": [
            "device",
            "criterion",
            "compute_per_task",
            "get_target_per_task",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        compute_per_task: dict[int, dict[Any, Callable]],
        get_target_per_task: dict[
            int, Callable[[dict[int, torch.Tensor], int], torch.Tensor]
        ],
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

    def _update_held_data(self, current_data: dict[str, Any]) -> None:
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

        self._update_held_data(current_data)

    @abstractmethod
    def report(self) -> None:
        """Report the evaluation results.

        Args:
            data (pd.DataFrame | torch.Tensor | list | dict): Evaluation results to report.
        """
        pass

    @classmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        try:
            validate(config, cls.json_schema)
        except ValidationError as e:
            logging.error(f"Config validation error: {e}")
            return False
        return True

    @classmethod
    def from_config(
        cls, config: dict[str, Any], apply_model: Callable | None = None
    ) -> "DefaultEvaluator":
        """Create DefaultEvaluator from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'device', 'criterion', 'compute_per_task', 'get_target_per_task', and 'apply_model'.

        Returns:
            DefaultEvaluator: An instance of DefaultEvaluator initialized with the provided configuration.
        """
        if not cls.verify_config(config):
            raise ValueError("Invalid configuration")

        device = config.get("device", "cpu")

        criterion = config.get("criterion", None)

        if criterion is None:
            raise ValueError("Criterion must be specified.")

        criterion_args = config.get("criterion_args", [])
        criterion_kwargs = config.get("criterion_kwargs", {})

        compute_per_task = config.get("compute_per_task", None)
        if compute_per_task is None:
            raise ValueError("Compute per task must be specified.")

        get_target_per_task = config.get("get_target_per_task", None)
        if get_target_per_task is None:
            raise ValueError("Get target per task must be specified.")

        device = config.get("device", "cpu")

        # build criterion
        try:
            criterion_type = utils.import_and_get(criterion)

            if criterion_type is None:
                raise ValueError(f"Failed to import criterion: {criterion}")
            criterion = criterion_type(*criterion_args, **criterion_kwargs)

        except Exception as e:
            raise ValueError(f"Failed to import criterion: {e}")

        # build per_task_monitors
        per_task_monitors = {}
        for key, cfg_list in compute_per_task.items():
            per_task_monitors[key] = []
            for cfg in cfg_list:
                try:
                    # try to find the type in the module in this module
                    try:
                        monitortype = globals().get(cfg["type"])
                    except Exception as e:
                        logging.warning(
                            f"Failed to find monitortype in globals for task {key}: {e}"
                        )

                        monitortype = utils.import_and_get(cfg["type"])

                    if monitortype is None:
                        logging.error(
                            f"Failed to import monitortype for task {key}: {cfg.get('type', None)}"
                        )
                        raise ValueError(
                            f"Failed to import monitortype for task {key}: {cfg.get('type', None)}"
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to import monitortype for task {key}: {e}"
                    )

                try:
                    metrics = monitortype.from_config(cfg)
                except Exception as e:
                    raise ValueError(f"Failed to create metric for task {key}: {e}")

                per_task_monitors[key].append(metrics)

        # try to build target extractors
        target_extractors = {}
        for key, cfg in get_target_per_task.items():
            try:
                try:
                    targetgetter = globals().get(cfg["type"])
                except Exception as e:
                    logging.warning(
                        f"Failed to find targetgetter in globals for task {key}: {e}"
                    )

                    targetgetter = utils.import_and_get(cfg["type"])

                if targetgetter is None:
                    raise ValueError(
                        f"Failed to import targetgetter for task {key}: {cfg.get('type', None)}"
                    )
            except Exception as e:
                raise ValueError(f"Failed to import targetgetter for task {key}: {e}")

            if not callable(targetgetter):
                raise ValueError(f"Target getter for task {key} is not callable")

            target_extractors[key] = targetgetter

        # try to build apply_model
        apply_model: Callable | None = None
        apply_model_cfg = config.get("apply_model", None)
        if apply_model_cfg is not None:
            try:
                apply_model_type = utils.import_and_get(apply_model_cfg["type"])
            except Exception as e:
                raise ValueError(f"Failed to import apply_model: {e}")
            try:
                apply_model_args = apply_model_cfg.get("args", [])
                apply_model_kwargs = apply_model_cfg.get("kwargs", {})
                if apply_model_type is not None:
                    logging.warning(
                        f"Trying to create apply_model from config {apply_model_cfg} but type is None"
                    )
                    apply_model = apply_model_type(
                        *apply_model_args, **apply_model_kwargs
                    )
            except Exception as e:
                raise ValueError(f"Failed to get apply_model args: {e}")

        return cls(
            device=device,
            criterion=criterion,
            compute_per_task=compute_per_task,
            get_target_per_task=target_extractors,
            apply_model=apply_model,
        )


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


class F1ScoreEval(base.Configurable):
    """F1Score evaluator, useful for evaluation of classification problems. A callable class that builds an f1 evaluator to a given set of specifications. Uses sklearn.metrics.f1_score for the computation fo the f1 score."""

    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "F1ScoreEval Configuration",
        "type": "object",
        "properties": {
            "average": {
                "type": "string",
                "enum": ["micro", "macro", "weighted", "none"],
                "description": "Averaging method for F1 score as used by scikit-learn.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of numerical labels to include in the F1 score computation.",
            },
        },
        "required": ["average"],
    }

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
    def verify_config(cls, config: dict[str, Any]) -> bool:
        """Verfiy the configuration via a json schema

        Args:
            config (dict[str, Any]): configuration to verify

        Returns:
            bool: True if the configuration is valid, False otherwise
        """

        try:
            validate(instance=config, schema=cls.json_schema)
        except ValidationError as e:
            logging.error(f"Config validation error: {e}")
            return False
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "F1ScoreEval":
        """Create F1ScoreEval from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'average', 'mode', and 'labels'.
        Returns:
            F1ScoreEval: An instance of F1ScoreEval initialized with the provided configuration.
        """
        if not cls.verify_config(config):
            raise ValueError("Invalid configuration for F1ScoreEval")

        return cls(
            average=config.get("average", "macro"),
            labels=config.get("labels", None),
        )


class AccuracyEval(base.Configurable):
    """Accuracy evaluator, primarily useful for evaluation of regression problems. A callable class that builds an accuracy evaluator to a given set of specifications."""

    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AccuracyEval Configuration",
        "type": "object",
        "properties": {
            "metrics": {
                "type": "string",
                "description": "The name of the metric function to use.",
            },
            "metrics_args": {
                "type": "array",
                "description": "The positional arguments (*args). Any JSON value type is permitted.",
                "items": {},
            },
            "metrics_kwargs": {
                "type": "object",
                "description": "The keyword arguments (**kwargs). Keys must be strings, values can be any JSON value type.",
                "additionalProperties": {},
            },
        },
        "required": ["metrics"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        metric: Callable | type[torch.nn.Module] | None,
        metrics_args: list[Any] = [],
        metrics_kwargs: dict[str, Any] = {},
    ) -> None:
        """Initialize a new AccuracyEvaluator.

        Args:
            metric (Callable | type[torch.nn.Module] | None): The metric function or model to use.
            metrics_args (list[Any], optional): Positional arguments for the metric function. Defaults to [].
            metrics_kwargs (dict[str, Any], optional): Keyword arguments for the metric function. Defaults to {}.
        """

        self.metrics: Callable | torch.nn.Module = torch.nn.MSELoss()

        if isinstance(metric, Callable):
            self.metrics = metric
        elif metric is not None and issubclass(metric, torch.nn.Module):
            self.metrics = metric(*metrics_args, **metrics_kwargs)
        else:
            # don't do anything here
            pass

    def __call__(
        self, data: dict[Any, Any], task: int
    ) -> float | np.float64 | np.ndarray | list:
        """Compute accuracy for a given task.

        Args:
            data (dict[Any, Any]): Dictionary containing outputs and targets.
            task (int): Task index.
        """
        y_true = torch.cat(data[f"target_{task}"])
        y_pred = torch.cat(data[f"output_{task}"])

        return self.metrics(y_pred, y_true).item()

    @classmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        """Verify configuration dict with a json schema

        Args:
            config (dict[str, Any]): Config to verify

        Raises:
            ValueError: If the config is invalid

        Returns:
            bool: True if the config is valid, False otherwise
        """

        try:
            validate(instance=config, schema=cls.json_schema)
        except ValidationError as e:
            logging.error(f"Configuration validation error for AccuracyEval: {e}")
            return False
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AccuracyEval":
        """Create AccuracyEval from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'average', 'mode', and 'labels'.
        Returns:
            AccuracyEval: An instance of AccuracyEval initialized with the provided configuration.
        """
        if not cls.verify_config(config):
            raise ValueError("Invalid configuration for AccuracyEval")

        metric = config.get("metrics", None)
        metricstype: Callable | type[torch.nn.Module] | None = None
        if metric is not None:
            try:
                metricstype = utils.import_and_get(metric)
            except ValueError as e:
                raise ValueError(
                    f"Could not import metrics {metric} for AccuracyEval"
                ) from e
        else:
            # nothing to be done here because by default we use MSELoss
            pass

        args = config.get("metrics_args", [])
        kwargs = config.get("metrics_kwargs", {})
        return cls(
            metric=metricstype,
            metrics_args=args,
            metrics_kwargs=kwargs,
        )


# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore
class DefaultEarlyStopping(base.Configurable):
    """Early stopping based on a validation metric."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AccuracyEval Configuration",
        "type": "object",
        "properties": {
            "tasks": {
                "type": "object",
                "description": "A list of properties for each task in the ML model to evaluate.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "delta": {
                            "type": float,
                            "description": "The minimum change in the metric to qualify as an improvement.",
                        },
                        "window": {
                            "type": int,
                            "description": "The window length in epochs when the data shall be smoothed.",
                            "minimum": 1,
                        },
                        "metric": {
                            "type": str,
                            "description": "The metric to optimize during training.",
                        },
                        "grace_period": {
                            "type": int,
                            "description": "The number of epochs with no improvement after which training will be stopped.",
                            "minimum": 0,
                        },
                        "patience": {
                            "type": int,
                            "description": "The number of epochs with no improvement after which training will be stopped.",
                            "minimum": 1,
                        },
                        "init_best_score": {
                            "type": float,
                            "description": "The initial best score for the task.",
                        },
                        "mode": {
                            "type": str,
                            "description": "Whether finding a better model min or max comparison based",
                            "enum": ["min", "max"],
                        },
                        "smoothed": {
                            "type": bool,
                            "description": "Whether to apply smoothing to the metric.",
                        },
                    },
                },
            },
            "mode": {
                "type": str,
                "description": "The mode for early stopping, either 'min' or 'max' or the name of a callable that can be imported.",
            },
        },
        "required": ["tasks", "mode"],
        "additionalProperties": False,
    }

    # put this into the package
    def __init__(
        self,
        tasks: dict[str | int, Any],
        mode: str | Callable[[dict[str | int, Any]], bool] = "any",
    ):
        """Instantiate a new DefaultEarlyStopping.

        Args:
            tasks (dict[str  |  int, Any]): dict of task definitions
            mode (str | Callable[[dict[str  |  int, Any]], bool], optional): Mode for aggregating evaluations. 'all', 'any', or a callable. Defaults to "any".
        """
        self.tasks = tasks

        for key, task in tasks.items():
            task["current_grace_period"] = task["grace_period"]
            task["current_patience"] = task["patience"]
            task["best_score"] = task["init_best_score"]
            task["found_better"] = False

        self.logger = logging.getLogger(__name__)
        self.mode = mode

    @property
    def found_better_model(self) -> bool:
        """Check if a better model has been found."""

        if self.mode == "any":
            return any(
                [
                    task["found_better"]
                    for task in self.tasks.values()
                    if task["found_better"]
                ]
            )
        elif self.mode == "all":
            return all(
                [
                    task["found_better"]
                    for task in self.tasks.values()
                    if task["found_better"]
                ]
            )
        elif callable(self.mode):
            return self.mode(self.tasks)
        else:
            raise ValueError("Mode must be 'any', 'all', or a callable in Evaluator")

    def reset(self, index: int | str = "all") -> None:
        """Reset all or a single task

        Args:
            index (int | str, optional): The index/key of the task to reset, or 'all' if all all tasks should be reset.. Defaults to "all".
        """
        if index == "all":
            for task in self.tasks.values():
                task["current_patience"] = task["patience"]
                task["found_better"] = False
                task["current_grace_period"] = task["grace_period"]
        else:
            task = self.tasks.get(index)
            if task is not None:
                task["current_patience"] = task["patience"]
                task["found_better"] = False
                task["current_grace_period"] = task["grace_period"]

    def add_task(
        self,
        index: int | str,
        delta: float,
        window: int,
        metric: str,
        grace_period: int,
        patience: int,
        smoothed: bool,
        init_best_score: float | int = np.inf,
        mode: str = "min",
    ) -> None:
        """Add a new task to the evaluator

        Args:
            index (int | str): The index/key of the task the new evaluation belongs to
            delta (float): Minimum change to consider an improvement.
            window (int): Size of the moving window for smoothing.
            metric (str): Metric to monitor for early stopping.
            grace_period (int): Grace period for early stopping.
            patience (int): Patience for early stopping.
            smoothed (bool): Whether to apply smoothing to the metric.
            init_best_score (float | int, optional): Initial best score for the metric. Defaults to np.inf.
            mode (str, optional): Mode for early stopping. Defaults to "min".
        """
        self.tasks[index] = {
            "delta": delta,
            "window": window,
            "metric": metric,
            "grace_period": grace_period,
            "current_grace_period": grace_period,
            "best_score": init_best_score,
            "found_better": False,
            "patience": patience,
            "current_patience": patience,
            "smoothed": smoothed,
        }

    def remove_task(self, index: int | str) -> None:
        """Remove a task from early stopping.

        Args:
            index (int | str): The index of the task to remove.
        """
        self.tasks.pop(index)

    def _evaluate_task(self, task: dict[str, Any], value: Any) -> bool:
        """Evaluate task

        Args:
            task (dict[str, Any]): The task to evaluate.
            value (Any): The value to compare against the task's best score.

        Raises:
            ValueError: If the task's mode is not recognized.

        Returns:
            bool: True if the task is improved, False otherwise.
        """
        if task["mode"] == "min":
            return task["best_score"] - task["delta"] > value
        elif task["mode"] == "max":
            return task["best_score"] - task["delta"] < value
        else:
            raise ValueError("Unknown value for mode in task. must be 'max' or 'min'.")
            return False

    def __call__(self, data: pd.DataFrame | pd.Series) -> bool:
        """Evaluate early stopping criteria. This is done by comparing the last value of data[self.metric] with the current best value recorded. If that value is better than the current best, the current best is updated,
        patience is reset and 'found_better' is set to True. Otherwise, if the number of datapoints in 'data' is greater than self.window, the patience is decremented.

        Args:
            data (pd.DataFrame | pd.Series): Recorded evaluation metrics in a pandas structure.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        for t in self.tasks.values():
            t["found_better"] = False
        ds = {}  # dict to hold current metric values

        # go over all registered metrics and check if the model performs better on any of them
        # then aggregrate the result with 'found_better_model'.
        for k in self.tasks.keys():
            # prevent a skipped metric from affecting early stopping
            if self.tasks[k]["metric"] not in data.columns:
                self.logger.warning(
                    f"    Metric {self.tasks[k]['metric']} not found in data."
                )
                self.tasks[k]["found_better"] = (
                    True  # treat it as if it has found a better model to avoid forever deadlock
                )
                continue

            if self.tasks[k]["smoothing"]:
                d = (
                    data[self.tasks[k]["metric"]]
                    .rolling(window=self.tasks[k]["window"], min_periods=1)
                    .mean()
                )
            else:
                d = data[self.tasks[k]["metric"]]

            better = self._evaluate_task(self.tasks[k], d.iloc[-1])

            if better and len(data) > 1:
                self.logger.info(
                    f"    Better model found at task {k}: {d.iloc[-1]:.8f}, current best: {self.tasks[k]['best_score']:.8f}"
                )
                self.tasks[k]["found_better"] = True

            ds[k] = d.iloc[-1]

        if self.found_better_model:
            # when we found a better model the stopping patience gets reset
            self.tasks[k]["current_patience"] = self.tasks[k][
                "patience"
            ]  # reset patience
            self.logger.info("Found better model")
            for j in self.tasks.keys():
                if self.tasks[j]["found_better"] and j in ds:
                    self.logger.info(
                        f"current best score: {self.tasks[j]['best_score']:.8f}, current score: {ds[j]:.8f}"
                    )
                    self.tasks[j]["best_score"] = ds[j]  # record best score

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

    @classmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        """Verify the configuraion file via a json schema.

        Args:
            config (dict[str, Any]): _description_

        Returns:
            bool: _description_
        """
        try:
            validate(config, cls.schema)
        except ValidationError as e:
            logging.error(f"Configuration validation error: {e}")
            return False
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefaultEarlyStopping":
        """Construct a DefaultEarlyStopping from a configuration dictionary.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
            ImportError: When the mode is not "min" or "max" and cannot be imported.
        Returns:
            DefaultEarlyStopping: The constructed DefaultEarlyStopping instance.
        """
        if cls.verify_config(config) is False:
            raise ValueError("Invalid configuration for Evaluator")

        tasks = config["tasks"]
        mode = config["mode"]

        # handle callable mode
        if mode not in ["min", "max"]:
            try:
                mode = utils.import_and_get(mode)
            except ImportError:
                logging.error(f"Could not import callable for mode: {mode}")
                raise ValueError(f"Invalid mode: {mode}")

        return cls(tasks, mode)
