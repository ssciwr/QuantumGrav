import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
import logging


class DefaultEvaluator:
    """Default evaluator for model evaluation - testing and validation during training"""

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
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
        self.data: pd.DataFrame | list = []
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> list[Any]:
        """Evaluate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to evaluate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for evaluation.

        Returns:
             list[Any]: A list of evaluation results.
        """
        model.eval()
        current_data = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data = batch.to(self.device)
                if self.apply_model:
                    outputs = self.apply_model(model, data)
                else:
                    outputs = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(outputs, data)
                current_data.append(loss)

        return current_data

    def report(self, data: list | pd.Series | torch.Tensor | np.ndarray) -> None:
        """Report the evaluation results.

        Args:
            data (list | pd.Series | torch.Tensor | np.ndarray): The evaluation results.
        """

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if isinstance(data, list):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.cpu().numpy()

        avg = np.mean(data)
        sigma = np.std(data)
        self.logger.info(f"Average loss: {avg}, Standard deviation: {sigma}")

        if isinstance(self.data, list):
            self.data.append((avg, sigma))
        else:
            self.data = pd.concat(
                [
                    self.data,
                    pd.DataFrame({"loss": avg, "std": sigma}, index=[0]),
                ],
                axis=0,
                ignore_index=True,
            )


class DefaultTester(DefaultEvaluator):
    """Default tester for model testing.

    Args:
        DefaultEvaluator (Class): Inherits from DefaultEvaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
    ):
        """Default tester for model testing.

        Args:
            device (str | torch.device | int,): The device to run the testing on.
            criterion (Callable): The loss function to use for testing.
            apply_model (Callable): A function to apply the model to the data.
        """
        super().__init__(device, criterion, apply_model)

    def test(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> list[Any]:
        """Test the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to test.
            data_loader (torch_geometric.loader.DataLoader): Data loader for testing.

        Returns:
            list[Any]: A list of testing results.
        """
        return self.evaluate(model, data_loader)


class DefaultValidator(DefaultEvaluator):
    """Default validator for model validation.

    Args:
        DefaultEvaluator (Class): Inherits from DefaultEvaluator and provides functionality for validating models
    using a specified criterion and optional model application function.
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
    ):
        """Default validator for model validation.

        Args:
            device (str | torch.device | int,): The device to run the validation on.
            criterion (Callable): The loss function to use for validation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
        """
        super().__init__(device, criterion, apply_model)

    def validate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> list[Any]:
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        Returns:
            list[Any]: A list of validation results.
        """
        return self.evaluate(model, data_loader)
