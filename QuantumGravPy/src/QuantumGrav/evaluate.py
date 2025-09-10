import torch
from typing import Callable, Any, Iterable
import torch_geometric
import numpy as np
import pandas as pd
import logging


class DefaultEvaluator:
    def __init__(
        self, device, criterion: Callable, apply_model: Callable | None = None
    ):
        """Default evaluator for model evaluation.

        Args:
            device (_type_): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable): A function to apply the model to the data.
        """
        self.criterion = criterion
        self.apply_model = apply_model
        self.device = device
        self.data = []
        self.logger = logging.getLogger(__name__)

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ) -> Any:
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
        """Report the evaluation results to stdout"""

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if isinstance(data, list):
            for i, d in enumerate(data):
                if isinstance(d, torch.Tensor):
                    data[i] = d.cpu().numpy()

        avg = np.mean(data)
        sigma = np.std(data)
        self.logger.info(f"Average loss: {avg}, Standard deviation: {sigma}")
        self.data.append((avg, sigma))


class DefaultTester(DefaultEvaluator):
    def __init__(
        self, device, criterion: Callable, apply_model: Callable | None = None
    ):
        """Default tester for model testing.

        Args:
            device (_type_): The device to run the testing on.
            criterion (Callable): The loss function to use for testing.
            apply_model (Callable): A function to apply the model to the data.
        """
        super().__init__(device, criterion, apply_model)

    def test(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ):
        """Test the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to test.
            data_loader (torch_geometric.loader.DataLoader): Data loader for testing.

        Returns:
            list[Any]: A list of testing results.
        """
        return self.evaluate(model, data_loader)


class DefaultValidator(DefaultEvaluator):
    def __init__(
        self, device, criterion: Callable, apply_model: Callable | None = None
    ):
        super().__init__(device, criterion, apply_model)

    def validate(
        self,
        model: torch.nn.Module,
        data_loader: torch_geometric.loader.DataLoader,  # type: ignore
    ):
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        Returns:
            list[Any]: A list of validation results.
        """
        return self.evaluate(model, data_loader)


class DefaultEarlyStopping:
    """Early stopping based on a validation metric."""

    def __init__(
        self,
        patience: int,
        delta: float = 1e-4,
        window=7,
    ):
        """Early stopping initialization.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float, optional): Minimum change to consider an improvement. Defaults to 1e-4.
            window (int, optional): Size of the moving window for smoothing. Defaults to 7.
        """
        self.patience = patience
        self.current_patience = patience
        self.delta = delta
        self.best_score = np.inf
        self.window = window
        self.found_better = False
        self.logger = logging.getLogger(__name__)

    def __call__(self, data: Iterable | pd.DataFrame | pd.Series) -> bool:
        """Check if early stopping criteria are met.

        Args:
            data: Iterable of validation metrics, e.g., list of scalars, list of tuples, Dataframe, numpy array...

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        window = min(self.window, len(data))
        smoothed = pd.Series(data).rolling(window=window, min_periods=1).mean()
        if smoothed.iloc[-1] < self.best_score - self.delta:
            self.logger.info(
                f"Early stopping patience reset: {self.current_patience} -> {self.patience}, early stopping best score updated: {self.best_score} -> {smoothed.iloc[-1]}"
            )
            self.best_score = smoothed.iloc[-1]
            self.current_patience = self.patience
            self.found_better = True
        else:
            self.logger.info(
                f"Early stopping patience decreased: {self.current_patience} -> {self.current_patience - 1}"
            )
            self.current_patience -= 1
            self.found_better = False

        return self.current_patience <= 0
