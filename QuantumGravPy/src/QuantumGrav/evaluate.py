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
