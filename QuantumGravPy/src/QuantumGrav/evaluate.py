import torch
from typing import Callable, Any, Sequence, Dict
import torch_geometric
import pandas as pd
import logging
import jsonschema
from abc import abstractmethod
import tqdm

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
                "description": "Flat list of monitor specs as objects: {name, monitor, args?, kwargs?}",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Metric name"},
                        "monitor": {
                            "description": "Callable or spec to construct the monitor"
                        },
                        "args": {
                            "type": "array",
                            "description": "Optional positional args",
                            "items": {},
                        },
                        "kwargs": {
                            "type": "object",
                            "description": "Optional keyword args",
                            "additionalProperties": {},
                        },
                    },
                    "required": ["name", "monitor"],
                    "additionalProperties": False,
                },
            },
            "apply_model": {
                "description": "Optional function to call the model's forward method in customized way"
            },
        },
        "required": ["device", "criterion"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        evaluator_tasks: Sequence[
            Dict[str, str | Callable | Sequence[Any] | None | Dict[str, Any] | None]
        ]
        | None,
        apply_model: Callable | None = None,
    ):
        """Default evaluator for model evaluation.

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            evaluator_tasks (Sequence[Dict[str, Any]]):
                Flat list of task objects, each with keys:
                - name: str — metric name/column label
                - monitor: Callable or constructor spec — function to compute the metric given predictions and targets
                - args (optional): list — positional arguments if `monitor` needs construction
                - kwargs (optional): dict — keyword arguments if `monitor` needs construction
                Tasks are not bound to specific output heads; each task can access all collected predictions/targets.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
        """
        self.criterion = criterion
        self.apply_model = apply_model
        self.device = device
        self.logger = logging.getLogger(__name__)

        # store as list of (metric_name, monitor_callable) tuples for simple iteration
        self.tasks: list[tuple[str, Callable]] = []
        columns = ["loss_avg", "loss_min", "loss_max"]

        if evaluator_tasks is not None:
            for task_spec in evaluator_tasks:
                monitor = task_spec["monitor"]
                if task_spec.get("args") or task_spec.get("kwargs"):
                    monitor = monitor(
                        *(task_spec.get("args", []) if task_spec.get("args") else []),
                        **(
                            task_spec.get("kwargs", {})
                            if task_spec.get("kwargs")
                            else {}
                        ),
                    )
                columns.append(task_spec["name"])
                self.tasks.append((task_spec["name"], monitor))

        self.data = pd.DataFrame({col: pd.Series([], dtype=object) for col in columns})

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
             pd.DataFrame: A dataframe containing the evaluation results.
        """
        model.eval()
        current_losses = []
        current_predictions = []
        current_targets = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(data_loader, desc="Evaluating")):
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
        for monitor_name, monitor in self.tasks:
            res = monitor(current_predictions, current_targets)
            if isinstance(res, torch.Tensor):
                res = res.cpu().item()
            self.data.loc[current_data_length, monitor_name] = res

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
            torch.device(config["device"]),
            config["criterion"],
            config.get("evaluator_tasks"),
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
        evaluator_tasks: Sequence[Dict[str, Any]],
        apply_model: Callable | None = None,
    ):
        """Default tester for model testing.

        Args:
            device (str | torch.device | int): The device to run the testing on.
            criterion (Callable): The loss function to use for testing.
            evaluator_tasks (Sequence[Dict[str, Any]]):
                List of task objects with keys {name, monitor, args?, kwargs?}. See `Evaluator.__init__` for details.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
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
        self.logger.info("\n%s", data.tail(1).to_string(float_format="{:.4f}".format))


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
        evaluator_tasks: Sequence[Dict[str, Any]],
        apply_model: Callable | None = None,
    ):
        """Default validator for model validation.

        Args:
            device (str | torch.device | int): The device to run the validation on.
            criterion (Callable): The loss function to use for validation.
            evaluator_tasks (Sequence[Dict[str, Any]]):
                List of task objects with keys {name, monitor, args?, kwargs?}. See `Evaluator.__init__` for details.
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
        self.logger.info("\n%s", data.tail(1).to_string(float_format="{:.4f}".format))
