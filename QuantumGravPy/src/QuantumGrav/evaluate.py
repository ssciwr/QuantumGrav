import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
import logging
from abc import abstractmethod
from sklearn.metrics import f1_score
from jsonschema import validate, ValidationError
from inspect import isclass

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
                "type": "object",
                "description": "The loss function to use for evaluation.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "the type name of the callable to compute the loss",
                    },
                    "args": {
                        "type": "array",
                        "description": "Optional positional arguments for the callable.",
                        "items": {},
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Optional keyword arguments for the callable.",
                        "additionalProperties": True,
                    },
                },
                "required": ["name"],
                "additional_properties": True,
            },
            "compute_per_task": {
                "type": "object",
                "description": "Task-specific metrics to compute.",
                "additionalProperties": {
                    "type": "array",
                    "description": "a list of {name, args, kwargs} dictionaries that define what metrics should be computed per task",
                    "items": {
                        "type": "object",
                        "description": "A single task definition",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "the type name of the callable to compute the desired quantity",
                            },
                            "name_in_data": {
                                "type": "string",
                                "description": "the name the result of this computation should have in the final dataframe",
                            },
                            "args": {
                                "type": "array",
                                "description": "Optional positional arguments for the callable.",
                                "items": {},
                            },
                            "kwargs": {
                                "type": "object",
                                "description": "Optional keyword arguments for the callable.",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["name"],
                        "additional_properties": True,
                    },
                },
            },
            "get_target_per_task": {
                "type": "object",
                "description": "Function to get the target for each task.",
                "additional_properties": True,
            },
            "apply_model": {
                "type": "object",
                "description": "Optional apply model function",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "the type name of the callable to compute the desired quantity",
                    },
                    "args": {
                        "type": "array",
                        "description": "Optional positional arguments for the callable.",
                        "items": {},
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Optional keyword arguments for the callable.",
                        "additionalProperties": True,
                    },
                },
                "required": ["name"],
                "additional_properties": True,
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
        """Initialize a new default evaluator

        Args:
            device (str | torch.device | int): device to work on
            criterion (Callable): Loss computation for the evaluation run
            compute_per_task (dict[int, dict[Any, Callable]]): Dictionary mapping task IDs to a set of named metrics to compute for each task
            get_target_per_task (dict[ int, Callable[[dict[int, torch.Tensor], int], torch.Tensor] ]): Function to get the target for each task
            apply_model (Callable | None, optional): Function to apply the model to the data. Defaults to None.
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

        self.active_tasks = set()
        for i, task in enumerate(model.active_tasks):
            if task:
                self.active_tasks.add(i)

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
                    y = self.get_target_per_task[i](data.y, i)
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

                # clean up outputs and targets
                del current_data[f"output_{i}"]
                del current_data[f"target_{i}"]

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
    def _find_function_or_class(cls, name: str) -> Any:
        """Helper method to find a function or class by name in globals or import it.

        Args:
            name (str): Name of the function or class to find.
        Returns:
            Any: The found function or class.
        """

        obj = utils.get_evaluation_function(name)

        if obj is None:
            try:
                obj = utils.import_and_get(name)
            except Exception as e:
                raise ValueError(f"Failed to import {name}: {e}")
        return obj

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefaultEvaluator":
        """Create DefaultEvaluator from configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary with keys 'device', 'criterion', 'compute_per_task', 'get_target_per_task', and 'apply_model'.

        Returns:
            DefaultEvaluator: An instance of DefaultEvaluator initialized with the provided configuration.
        """
        if not cls.verify_config(config):
            raise ValueError("Invalid configuration")

        device = config.get("device", "cpu")

        criterion_cfg = config.get("criterion", None)
        if criterion_cfg is None:
            raise ValueError("Criterion must be specified.")

        criterion_type = criterion_cfg["name"]
        criterion_args = criterion_cfg.get("args", [])
        criterion_kwargs = criterion_cfg.get("kwargs", {})

        compute_per_task = config.get("compute_per_task", None)
        if compute_per_task is None:
            raise ValueError("Compute per task must be specified.")

        get_target_per_task = config.get("get_target_per_task", None)
        if get_target_per_task is None:
            raise ValueError("Get target per task must be specified.")

        device = config.get("device", "cpu")

        # build criterion
        try:
            criterion_type = cls._find_function_or_class(criterion_type)
            if criterion_type is None:
                raise ValueError(f"Failed to import criterion: {criterion_type}")
            if isclass(criterion_type):
                criterion = criterion_type(*criterion_args, **criterion_kwargs)
            else:
                criterion = criterion_type
        except Exception as e:
            raise ValueError(f"Failed to import criterion: {e}")

        # build per_task_monitors. each task can have a dict of named comput tasks. in the config,
        # these can refer to plain callables or classes to be instantiated
        per_task_monitors = {}
        for key, cfg_list in compute_per_task.items():
            per_task_monitors[int(key)] = {}
            for cfg in cfg_list:
                # try to find the type in the module in this module
                monitortype = cls._find_function_or_class(cfg["name"])
                name_in_data = cfg["name_in_data"]
                if monitortype is None:
                    raise ValueError(
                        f"Failed to import monitortype for task {key}: {cfg['name']}"
                    )

                if isclass(monitortype):
                    try:
                        metrics = monitortype.from_config(cfg)
                    except Exception as _:
                        try:
                            metrics = monitortype(
                                *cfg.get("args", []), **cfg.get("kwargs", {})
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Failed to build monitortype for task {key} from config: {e}"
                            )
                elif callable(monitortype):
                    metrics = monitortype
                else:
                    raise ValueError(
                        f"Monitortype for task {key} is not a class or callable"
                    )
                per_task_monitors[int(key)][name_in_data] = metrics

        # try to build target extractors
        target_extractors = {}
        for key, funcname in get_target_per_task.items():
            targetgetter = cls._find_function_or_class(funcname)
            if targetgetter is None:
                raise ValueError(
                    f"Failed to import targetgetter for task {key}: {funcname}"
                )

            if not callable(targetgetter):
                raise ValueError(f"Target getter for task {key} is not callable")

            target_extractors[int(key)] = targetgetter

        # try to build apply_model
        apply_model: Callable | None = None
        apply_model_cfg = config.get("apply_model", None)
        if apply_model_cfg is not None:
            apply_model_type = cls._find_function_or_class(apply_model_cfg["name"])

            if apply_model_type is None:
                raise ValueError(
                    f"Failed to import apply_model: {apply_model_cfg['name']}"
                )

            if isclass(apply_model_type):
                apply_model_args = apply_model_cfg.get("args", [])
                apply_model_kwargs = apply_model_cfg.get("kwargs", {})
                apply_model = apply_model_type(*apply_model_args, **apply_model_kwargs)
            elif callable(apply_model_type):
                apply_model = apply_model_type
            else:
                raise ValueError(
                    f"apply_model type is not a class or callable: {apply_model_type}"
                )

        return cls(
            device=torch.device(device),
            criterion=criterion,
            compute_per_task=per_task_monitors,
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
        "additionalProperties": True,
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

    def __call__(self, data: dict[Any, Any], task: int) -> float | np.ndarray:
        """Compute F1 score for a given task.

        Args:
            data (dict[Any, Any]): Dictionary containing outputs and targets.
            task (int): Task index.
        """
        # if the model output has more than one dimension, we need to adjust the y_pred accordingly
        # and squeeze/unsqueeze as needed
        if f"target_{task}" not in data or f"output_{task}" not in data:
            raise KeyError(f"Task {task} not found in data for F1ScoreEval")

        y_true = torch.cat(data[f"target_{task}"])
        y_pred = torch.cat(data[f"output_{task}"])
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

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
            raise ValidationError("Invalid configuration for F1ScoreEval")

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
            "metric": {
                "type": "string",
                "description": "The name of the metric function to use.",
            },
            "metric_args": {
                "type": "array",
                "description": "The positional arguments (*args). Any JSON value type is permitted.",
                "items": {},
            },
            "metric_kwargs": {
                "type": "object",
                "description": "The keyword arguments (**kwargs). Keys must be strings, values can be any JSON value type.",
                "additionalProperties": {},
            },
        },
        "required": ["metric"],
        "additionalProperties": True,
    }

    def __init__(
        self,
        metric: Callable | type[torch.nn.Module] | None,
        metric_args: list[Any] = [],
        metric_kwargs: dict[str, Any] = {},
    ) -> None:
        """Initialize a new AccuracyEvaluator.

        Args:
            metric (Callable | type[torch.nn.Module] | None): The metric function or model to use.
            metric_args (list[Any], optional): Positional arguments for the metric function. Defaults to [].
            metric_kwargs (dict[str, Any], optional): Keyword arguments for the metric function. Defaults to {}.
        """
        if metric is None:
            self.metric: Callable | torch.nn.Module = torch.nn.MSELoss()
        elif isinstance(metric, Callable) and not isclass(metric):
            self.metric = metric
        elif isclass(metric):
            self.metric = metric(*metric_args, **metric_kwargs)
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
        if f"target_{task}" not in data or f"output_{task}" not in data:
            raise KeyError(f"Missing data for task {task} in AccuracyEval")

        y_true = torch.cat(data[f"target_{task}"])
        y_pred = torch.cat(data[f"output_{task}"])

        return self.metric(y_pred, y_true).item()

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
            raise ValidationError("Invalid configuration for AccuracyEval")

        metric = config.get("metric", None)
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

        args = config.get("metric_args", [])
        kwargs = config.get("metric_kwargs", {})
        return cls(
            metric=metricstype,
            metric_args=args,
            metric_kwargs=kwargs,
        )


# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore
class DefaultEarlyStopping(base.Configurable):
    """Early stopping based on a validation metric."""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DefaultEarlyStopping Configuration",
        "type": "object",
        "properties": {
            "tasks": {
                "type": "object",
                "description": "A list of properties for each task in the ML model to evaluate.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "delta": {
                            "type": "number",
                            "description": "The minimum change in the metric to qualify as an improvement.",
                        },
                        "metric": {
                            "type": "string",
                            "description": "The metric to optimize during training.",
                        },
                        "grace_period": {
                            "type": "integer",
                            "description": "The number of epochs with no improvement after which training will be stopped.",
                            "minimum": 0,
                        },
                        "init_best_score": {
                            "type": "number",
                            "description": "The initial best score for the task.",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Whether finding a better model min or max comparison based",
                            "enum": ["min", "max"],
                        },
                    },
                    "additionalProperties": False,
                    "required": [
                        "delta",
                        "metric",
                        "grace_period",
                        "init_best_score",
                        "mode",
                    ],
                },
            },
            "mode": {
                "type": "string",
                "description": "How to aggregate the results of the different tasks - either 'any' (improvement in any one task will be seen as positive and trigger patience reset) or 'all' (improvement in all tasks is required)",
                "enum": ["any", "all"],
            },
            "patience": {
                "type": "integer",
                "description": "How many epochs to wait before training stops when no better model can be found",
            },
        },
        "required": ["tasks", "mode"],
        "additionalProperties": False,
    }

    # put this into the package
    def __init__(
        self,
        tasks: dict[str | int, Any],
        patience: int,
        mode: str = "any",
    ):
        """Instantiate a new DefaultEarlyStopping.

        Args:
            tasks (dict[str  |  int, Any]): dict of task definitions
            patience (int): how long to wait until early stopping is triggered if no better model can be found.
            mode (str | Callable[[dict[str  |  int, Any]], bool], optional): Mode for aggregating evaluations. 'all', 'any', or a callable. Defaults to "any".
        """
        self.tasks = tasks

        for _, task in tasks.items():
            task["current_grace_period"] = task["grace_period"]
            task["best_score"] = task["init_best_score"]
            task["found_better"] = False

        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.patience = patience
        self.current_patience = patience

    @property
    def grace_periods_ran_out(self) -> bool:
        """Check if all grace periods have run out"""
        return all(
            task["current_grace_period"] <= task["grace_period"]
            for task in self.tasks.values()
        )

    @property
    def found_better_model(self) -> bool:
        """Check if a better model has been found."""
        status = [task["found_better"] for task in self.tasks.values()]
        if self.mode == "any":
            return any(status)
        elif self.mode == "all":
            return all(status)
        else:
            raise ValueError("Mode must be 'any', 'all'")

    def reset(self, index: int | str = "all") -> None:
        """Reset all or a single task

        Args:
            index (int | str, optional): The index/key of the task to reset, or 'all' if all all tasks should be reset.. Defaults to "all".
        """
        if index == "all":
            for task in self.tasks.values():
                task["found_better"] = False
                task["current_grace_period"] = task["grace_period"]
        else:
            task = self.tasks.get(index)
            if task is not None:
                task["found_better"] = False
                task["current_grace_period"] = task["grace_period"]

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
            # smooth data if desired - prevents oscillations
            d = data[self.tasks[k]["metric"]]

            better = self._evaluate_task(self.tasks[k], d.iloc[-1])

            if better and len(data) >= 1:
                self.logger.debug(
                    f"    Better model found for task {k}: {d.iloc[-1]:.8f}, current best: {self.tasks[k]['best_score']:.8f}"
                )
                self.tasks[k]["found_better"] = True

            ds[k] = d.iloc[-1]

        if self.found_better_model:
            self.logger.info("Found better model")
            for j in self.tasks.keys():
                if self.tasks[j]["found_better"] and j in ds:
                    self.logger.info(
                        f"task {j}: current best score: {self.tasks[j]['best_score']:.8f}, current score: {ds[j]:.8f}"
                    )
                    self.tasks[j]["best_score"] = ds[j]  # record best score

                # when we found a better model the stopping patience gets reset
                self.current_patience = self.patience

        # only when all grace periods are done will we reduce the patience
        elif self.grace_periods_ran_out:
            self.current_patience -= 1
        else:
            pass  # do nothing here, grace period needs to go down anyway

        # reduce the grace period
        for task in self.tasks.values():
            if task["current_grace_period"] > 0:
                task["current_grace_period"] -= 1

        # general reporting
        for k, t in self.tasks.items():
            self.logger.info(
                f"EarlyStopping task {k}: current patience: {self.current_patience}, best score: {self.tasks[k]['best_score']:.8f}, grace_period: {self.tasks[k]['current_grace_period']}"
            )

        return self.current_patience <= 0

    @classmethod
    def verify_config(cls, config: dict[str, Any]) -> bool:
        """Verify the configuraion file via a json schema.

        Args:
            config (dict[str, Any]): config to verify

        Returns:
            bool: Whether or not the config adheres to the defined schema
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
            ValidationError: If the configuration is invalid.
            ImportError: When the mode is not "min" or "max" and cannot be imported.
        Returns:
            DefaultEarlyStopping: The constructed DefaultEarlyStopping instance.
        """
        if cls.verify_config(config) is False:
            raise ValidationError("Invalid configuration for Evaluator")

        tasks = config["tasks"]
        mode = config["mode"]
        patience = config["patience"]

        return cls(tasks, patience, mode)
