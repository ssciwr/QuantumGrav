import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
import logging


class DefaultEvaluator:
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
    """Default validator for model validation.

    Args:
        DefaultEvaluator (_type_): _description_
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
    ):
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        Returns:
            list[Any]: A list of validation results.
        """
        return self.evaluate(model, data_loader)


# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore
class DefaultEarlyStopping:
    """Early stopping based on a validation metric."""

    # put this into the package
    def __init__(
        self,
        patience: int,
        delta: float = 1e-4,
        window=7,
        metric: str = "loss",
        smoothing: bool = False,
        criterion: Callable = lambda early_stopping_instance,
        data: early_stopping_instance.current_patience <= 0,
        init_best_score: float = np.inf,
    ):
        """Early stopping initialization.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float, optional): Minimum change to consider an improvement. Defaults to 1e-4.
            window (int, optional): Size of the moving window for smoothing. Defaults to 7.
            metric (str, optional): Metric to monitor for early stopping. Defaults to "loss".
            smoothing (bool, optional): Whether to apply smoothed mean to the metric to dampen fluctuations. Defaults to False.
            criterion (Callable, optional): Custom stopping criterion. Defaults to a function that stops when patience is exhausted.
        """
        self.patience = patience
        self.current_patience = patience
        self.delta = delta
        self.window = window
        self.found_better = False
        self.metric = metric
        self.best_score = init_best_score
        self.smoothing = smoothing
        self.logger = logging.getLogger(__name__)
        self.criterion = criterion

    def __call__(self, data: pd.DataFrame | pd.Series) -> bool:
        """Evaluate early stopping criteria. This is done by comparing the last value of data[self.metric] with the current best value recorded. If that value is better than the current best, the current best is updated,
        patience is reset and 'found_better' is set to True. Otherwise, if the number of datapoints in 'data' is greater than self.window, the patience is decremented.

        Args:
            data (pd.DataFrame | pd.Series): Recorded evaluation metrics in a pandas structure.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        self.found_better = False

        if self.smoothing:
            d = data[self.metric].rolling(window=self.window, min_periods=1).mean()
        else:
            d = data[self.metric]

        if self.best_score - self.delta > d.iloc[-1]:  # always minimize the metric
            self.logger.info(
                f"    Better model found: {d.iloc[-1]:.8f}, current best: {self.best_score:.8f}"
            )
            self.best_score = d.iloc[-1]  # record best score
            self.current_patience = self.patience  # reset patience
            self.found_better = True
        elif len(data) > self.window:
            self.current_patience -= 1
        else:
            pass
            # don't do anything here, we want at least 'window' many epochs before patience is reduced

        self.logger.info(
            f"EarlyStopping: current patience: {self.current_patience}, best score: {self.best_score:.8f}"
        )

        return self.criterion(self, data)
