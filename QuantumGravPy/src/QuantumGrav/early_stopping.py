from typing import Any, Dict
import pandas as pd
import logging
from jsonschema import validate
from copy import deepcopy
from . import base


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
                            "minimum": 0.0,
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
                "minimum": 0,
            },
        },
        "required": ["tasks", "mode", "patience"],
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
        return all(task["current_grace_period"] <= 0 for task in self.tasks.values())

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
            index (int | str, optional): The index/key of the task to reset, or 'all' if all tasks should be reset. Defaults to "all".
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
        """Verify the configuration file via a json schema.

        Args:
            config (dict[str, Any]): config to verify

        Returns:
            bool: Whether or not the config adheres to the defined schema
        """
        validate(config, cls.schema)

        # Returns True only if validate does not raise a ValidationError
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DefaultEarlyStopping":
        """Construct a DefaultEarlyStopping from a configuration dictionary.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValidationError: If the configuration is invalid.
        Returns:
            DefaultEarlyStopping: The constructed DefaultEarlyStopping instance.
        """
        cls.verify_config(config)
        conf = deepcopy(config)

        tasks = conf["tasks"]
        mode = conf["mode"]
        patience = conf["patience"]

        return cls(tasks, patience, mode)

    def to_config(self) -> Dict[Any, Any]:
        """Build a config dictionary from the caller instance

        Returns:
            Dict[Any, Any]: Config representation of the caller instance
        """
        conf = {
            "tasks": {
                key: {
                    "delta": value["delta"],
                    "metric": value["metric"],
                    "grace_period": value["grace_period"],
                    "init_best_score": value["init_best_score"],
                    "mode": value["mode"],
                }
                for key, value in self.tasks.items()
            },
            "mode": self.mode,
            "patience": self.patience,
        }

        return conf
