import torch
from typing import Callable, Any, Tuple, Sequence, Dict
import torch_geometric
import pandas as pd
import logging
import jsonschema
from abc import abstractmethod

from . import base


class Evaluator(base.Configurable):
    """Default evaluator for model evaluation - testing and validation during training"""

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DefaultEarlyStopping Configuration",
        "type": "object",
        "properties": {
            "device": {"type": "string", "description": "The device to work on"},
            "criterion": {"desccription": "Loss function for the model evaluation"},
            "evaluator_tasks": {
                "type": "array",
                "description": "Sequence[Sequence[Tuple[str, Callable]]] - nested sequence of metric tasks",
                "items": {
                    "type": "array",
                    "description": "Sequence of (metric_name, callable) tuples",
                    "items": {
                        "type": "array",
                        "description": "Tuple of [metric_name, callable]",
                        "minItems": 2,
                        "maxItems": 2,
                        "prefixItems": [
                            {"type": "string", "description": "Metric name"},
                            {
                                "type": {
                                    "description": "Fully-qualified import path or name of the type/callable to initialize",
                                },
                                "args": {
                                    "type": "array",
                                    "description": "Positional arguments for constructor",
                                    "items": {},
                                },
                                "kwargs": {
                                    "type": "object",
                                    "description": "Keyword arguments for constructor",
                                    "additionalProperties": {},
                                },
                            },
                        ],
                    },
                },
                "additionalProperties": True,
            },
            "apply_model": {
                "description": "Optional function to call the model's forward method in customized way"
            },
        },
        "required": ["device", "criterion", "evaluator_tasks"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        evaluator_tasks: Sequence[
            Sequence[Tuple[str, Callable, Sequence[Any] | None, Dict[str, Any] | None]]
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
        self.logger = logging.getLogger(__name__)

        self.tasks: Dict[int, Dict[str, Callable]] = {}
        columns = ["loss_avg", "loss_min", "loss_max"]
        for task_id, per_task_monitors in enumerate(evaluator_tasks):
            for name, monitor, args, kwargs in per_task_monitors:
                if args or kwargs:
                    monitor = monitor(
                        *(args if args else []), **(kwargs if kwargs else {})
                    )
                columns.append(f"{name}_{task_id}")
                tasks = self.tasks.get(task_id, {})
                tasks[name] = monitor
                self.tasks[task_id] = tasks

        self.data = pd.DataFrame({col: [] for col in columns})

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> pd.DataFrame:
        """Evaluate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to evaluate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for evaluation.

        Returns:
             list[Any]: A list of evaluation results.
        """
        model.eval()
        current_losses = []
        current_predictions = []
        current_targets = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data = batch.to(self.device)
                if self.apply_model:
                    outputs = self.apply_model(model, data)
                else:
                    outputs = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(outputs, data)
                current_losses.append(loss.unsqueeze(0))
                current_predictions.append(outputs)
                current_targets.append(data.y)

        current_data_length = len(self.data)

        # tasks are not associated by default to any specific output head,
        # so we run all tasks on the collected outputs and targets
        for task_id, task_monitor_dict in self.tasks.items():
            for monitor_name, monitor in task_monitor_dict.items():
                colname = f"{monitor_name}_{task_id}"
                res = monitor(current_predictions, current_targets)
                if isinstance(res, torch.Tensor):
                    res = res.cpu().item()

                self.data.loc[current_data_length, colname] = res

        t_current_losses = torch.cat(current_losses).cpu()
        self.data.loc[current_data_length, "loss_avg"] = t_current_losses.mean().item()
        self.data.loc[current_data_length, "loss_min"] = t_current_losses.min().item()
        self.data.loc[current_data_length, "loss_max"] = t_current_losses.max().item()
        return self.data

    @abstractmethod
    def report(self, data: pd.DataFrame) -> None:
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Evaluator":
        """Build the evaluator from a config dictionary

        Args:
            config (Dict[str, Any]): Config dictionary to build the evaluator from.

        Returns:
            Evaluator: new evaluator instance
        """
        jsonschema.validate(config, cls.schema)
        return cls(
            config["device"],
            config["criterion"],
            config["evaluator_tasks"],
            config.get("apply_model"),
        )


class Tester(Evaluator):
    """Default tester for model testing.

    Args:
        Evaluator (Class): Inherits from Evaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        evaluator_tasks: Sequence[
            Sequence[Tuple[str, Callable, Sequence[Any] | None, Dict[str, Any] | None]]
        ],
        apply_model: Callable | None = None,
    ):
        """Default tester for model testing.

        Args:
            device (str | torch.device | int,): The device to run the testing on.
            criterion (Callable): The loss function to use for testing.
            apply_model (Callable): A function to apply the model to the data.
        """
        super().__init__(device, criterion, evaluator_tasks, apply_model)

    def test(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> pd.DataFrame:
        """Test the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to test.
            data_loader (torch_geometric.loader.DataLoader): Data loader for testing.

        Returns:
            pd.DataFrame: the current test results
        """
        return self.evaluate(model, data_loader)

    def report(self, data: pd.DataFrame) -> None:
        """Report the monitoring data"""
        self.logger.info("Testing results: ")
        self.logger.info(f" {data.tail(1)}")


class Validator(Evaluator):
    """Default validator for model validation.

    Args:
        Evaluator (Class): Inherits from Evaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        evaluator_tasks: Sequence[
            Sequence[Tuple[str, Callable, Sequence[Any] | None, Dict[str, Any] | None]]
        ],
        apply_model: Callable | None = None,
    ):
        """Default validator for model validation.

        Args:
            device (str | torch.device | int,): The device to run the validation on.
            criterion (Callable): The loss function to use for validation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
        """
        super().__init__(device, criterion, evaluator_tasks, apply_model)

    def validate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> pd.DataFrame:
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        Returns:
            pd.DataFrame: the current validation result
        """
        return self.evaluate(model, data_loader)

    def report(self, data: pd.DataFrame) -> None:
        """Report the monitoring data"""
        self.logger.info("Validation results: ")
        self.logger.info(f" {data.tail(1)}")
