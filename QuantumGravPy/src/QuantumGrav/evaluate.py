import torch
from typing import Callable, Any
import torch_geometric
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import logging
import tqdm


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
        criterion: Callable = lambda x, d: x.current_patience <= 0,
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

        if self.best_score + self.delta > d.iloc[-1]:
            self.logger.info(
                f"    Better model found: {d.iloc[-1]}, current best: {self.best_score}"
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
            f"EarlyStopping: current patience: {self.current_patience}, best score: {self.best_score}"
        )

        return self.criterion(self, data)


# Testing and Validation helper classes
class F1Evaluator(DefaultEvaluator):
    """Basic evaluator for classification tasks using F1 score.

    Args:
        DefaultEvaluator: Default evaluator class
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
        prefix: str = "",
    ):
        """Instantiate the F1 evaluator. This class uses the F1 score as the main metric for evaluation,
        and hence is mostly suited for classification tasks.

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
            prefix (str, optional): A prefix to add to the evaluation metrics. Defaults to "".

        Returns:
            _type_: _description_
        """
        super().__init__(device, criterion, apply_model)
        self.prefix = prefix
        self.data = pd.DataFrame(
            columns=[
                "avg_loss",
                "std_loss",
                "f1_per_class",
                "f1_unweighted",
                "f1_weighted",
                "f1_micro",
            ],
        )

    def evaluate(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:  # type: ignore
        """Evaluate the model on the given data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch_geometric.data.DataLoader): The data loader for the evaluation data.

        Returns:
            pd.DataFrame: The evaluation results.
        """
        model.eval()
        losses = []
        target = []
        output = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(data_loader, desc="Validation")):
                data = batch.to(self.device)

                if self.apply_model:
                    predictions = self.apply_model(model, data)
                else:
                    predictions = model(data.x, data.edge_index, data.batch)

                loss = self.criterion(predictions, data)

                losstensor = torch.full(
                    (batch.num_graphs,), loss, dtype=torch.float32
                ).cpu()

                losses.append(losstensor)

                if data.y.ndim == 2:
                    target.append(data.y[:, 0])
                elif data.y.ndim == 1:
                    target.append(data.y.cpu())
                else:
                    raise ValueError(f"Unexpected target shape: {data.y.shape}")
                outputs = ((torch.sigmoid(predictions.squeeze()) > 0.5).long()).cpu()
                output.append(outputs)

                if loss.isnan().any():
                    self.logger.warning(f"NaN loss encountered in batch {i}.")
                    continue

                if np.isnan(output).any():
                    self.logger.warning(f"NaN encountered in output in batch {i}.")
                    continue

                if torch.isnan(data.y).any():
                    self.logger.warning(f"NaN target encountered in batch {i}.")
                    continue

        current_data = pd.DataFrame(
            {
                "loss": torch.cat(losses),
                "output": torch.cat(output),
                "target": torch.cat(target),
            }
        )

        # compute data statistics and performance eval
        per_class = f1_score(
            current_data["target"], current_data["output"], average=None
        )
        unweighted = f1_score(
            current_data["target"], current_data["output"], average="macro"
        )
        weighted = f1_score(
            current_data["target"], current_data["output"], average="weighted"
        )
        micro = f1_score(
            current_data["target"], current_data["output"], average="micro"
        )
        avg_loss = current_data["loss"].mean()
        std_loss = current_data["loss"].std()

        self.data.loc[len(self.data)] = [
            avg_loss,
            std_loss,
            per_class,
            unweighted,
            weighted,
            micro,
        ]

        return current_data

    def report(self, data: pd.DataFrame | dict) -> None:
        """Report the evaluation results.

        Args:
            data (pd.DataFrame | dict): The evaluation data.
        """

        avg_loss = self.data["avg_loss"].iloc[-1]
        std_loss = self.data["std_loss"].iloc[-1]
        per_class = self.data["f1_per_class"].iloc[-1]
        unweighted = self.data["f1_unweighted"].iloc[-1]
        weighted = self.data["f1_weighted"].iloc[-1]
        micro = self.data["f1_micro"].iloc[-1]

        self.logger.info(f"{self.prefix} avg loss: {avg_loss:.4f} +/- {std_loss:.4f}")
        self.logger.info(f"{self.prefix} f1 score per class: {per_class}")
        self.logger.info(f"{self.prefix} f1 score unweighted: {unweighted}")
        self.logger.info(f"{self.prefix} f1 score weighted: {weighted}")
        self.logger.info(f"{self.prefix} f1 score micro: {micro}")


# for validation we reuse part of the tester class
class F1Validator(F1Evaluator):
    """F1 Score Validator

    Args:
        F1Evaluator (class): Base class for F1 score evaluation.
    """

    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        """F1 Score Validator

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
        """
        super().__init__(device, criterion, apply_model, prefix="Validation")

    def validate(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): The model to validate.
            data_loader (torch_geometric.data.DataLoader): The data loader for the validation set.

        Returns:
            pd.DataFrame: The validation results.
        """
        return super().evaluate(model, data_loader)


class F1Tester(F1Evaluator):
    """F1 Score Tester

    Args:
        F1Evaluator (class): Base class for F1 score evaluation.
    """

    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        """F1 Score Tester

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
        """
        super().__init__(device, criterion, apply_model, prefix="Test")

    def test(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:
        """Test the model on the given data loader.

        Args:
            model (torch.nn.Module): The model to test.
            data_loader (torch_geometric.data.DataLoader): The data loader for the test set.

        Returns:
            pd.DataFrame: The test results.
        """
        return super().evaluate(model, data_loader)


class AccuracyEvaluator(DefaultEvaluator):
    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
        prefix: str = "",
    ):
        """Instantiate the Accuracy evaluator. This class uses the accuracy as the main metric for evaluation,
        and hence is mostly suited for classification tasks.

        Args:
            device (str | torch.device | int): The device to run the evaluation on.
            criterion (Callable): The loss function to use for evaluation.
            apply_model (Callable | None, optional): A function to apply the model to the data. Defaults to None.
            prefix (str, optional): A prefix to add to the evaluation metrics. Defaults to "".

        Returns:
            _type_: _description_
        """
        super().__init__(device, criterion, apply_model)
        self.prefix = prefix
        self.data = pd.DataFrame(
            columns=[
                "avg_loss",
                "std_loss",
                "accuracy",
                "mse",
                "mae",
            ],
        )

    def evaluate(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:  # type: ignore
        """Evaluate the model on the given data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch_geometric.data.DataLoader): The data loader for the evaluation set.

        Raises:
            ValueError: If the evaluation fails.

        Returns:
            pd.DataFrame: The evaluation results.
        """
        model.eval()
        losses = []
        target = []
        output = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(data_loader, desc="Validation")):
                data = batch.to(self.device)

                if self.apply_model:
                    predictions = self.apply_model(model, data)
                else:
                    predictions = model(data.x, data.edge_index, data.batch)

                loss = self.criterion(predictions, data)

                losstensor = torch.full(
                    (batch.num_graphs,), loss, dtype=torch.float32
                ).cpu()

                losses.append(losstensor)

                if data.y.ndim == 2:
                    target.append(data.y[:, 0])
                elif data.y.ndim == 1:
                    target.append(data.y.cpu())
                else:
                    raise ValueError(f"Unexpected target shape: {data.y.shape}")

                outputs = ((torch.sigmoid(predictions.squeeze()) > 0.5).long()).cpu()
                output.append(outputs)

                if loss.isnan().any():
                    self.logger.warning(f"NaN loss encountered in batch {i}.")
                    continue

                if np.isnan(output).any():
                    self.logger.warning(f"NaN encountered in output in batch {i}.")
                    continue

                if torch.isnan(data.y).any():
                    self.logger.warning(f"NaN target encountered in batch {i}.")
                    continue

        output = torch.cat(output).to(torch.float32)
        target = torch.cat(target).to(torch.float32)
        current_data = pd.DataFrame(
            {
                "loss": torch.cat(losses),
                "output": output,
                "target": target,
            }
        )
        avg_loss = current_data["loss"].mean()
        std_loss = current_data["loss"].std()
        accuracy = (output == target).float().mean().item()
        mse = torch.nn.functional.mse_loss(output, target).item()
        mae = torch.nn.functional.l1_loss(output, target).item()

        self.data.loc[len(self.data)] = [avg_loss, std_loss, accuracy, mse, mae]

        return current_data

    def report(self, _: pd.DataFrame | dict):
        """Report the evaluation results.

        Args:
            _ (pd.DataFrame | dict): The evaluation results.
        """
        avg_loss = self.data["avg_loss"].iloc[-1]
        std_loss = self.data["std_loss"].iloc[-1]
        accuracy = self.data["accuracy"].iloc[-1]
        mse = self.data["mse"].iloc[-1]
        mae = self.data["mae"].iloc[-1]

        self.logger.info(f"{self.prefix} avg loss: {avg_loss:.4f} +/- {std_loss:.4f}")
        self.logger.info(f"{self.prefix} accuracy: {accuracy:.4f}")
        self.logger.info(f"{self.prefix} mse: {mse:.4f}")
        self.logger.info(f"{self.prefix} mae: {mae:.4f}")


class AccuracyValidator(AccuracyEvaluator):
    """Validate the model on the validation set.

    Args:
        AccuracyEvaluator (_type_): _description_
    """

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
    ):
        """Instantiate a new AccuracyValidator

        Args:
            device (str | torch.device | int): The device to use.
            criterion (Callable): The loss function to use.
            apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
        """
        super().__init__(device, criterion, apply_model, prefix="Validation")

    def validate(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:
        """Validate the model on the validation set.

        Args:
            model (torch.nn.Module): The model to validate.
            data_loader (torch_geometric.data.DataLoader): The data loader for the validation set.

        Returns:
            pd.DataFrame: The evaluation results.
        """
        return super().evaluate(model, data_loader)


class AccuracyTester(AccuracyEvaluator):
    """Test the model on the test set."""

    def __init__(
        self,
        device: str | torch.device | int,
        criterion: Callable,
        apply_model: Callable | None = None,
    ):
        """Instantiate a new AccuracyTester

        Args:
            device (str | torch.device | int): The device to use.
            criterion (Callable): The loss function to use.
            apply_model (Callable | None, optional): A function to apply the model. Defaults to None.
        """
        super().__init__(device, criterion, apply_model, prefix="Test")

    def test(
        self, model: torch.nn.Module, data_loader: torch_geometric.data.DataLoader
    ) -> pd.DataFrame:
        """Test the model on the test set.

        Args:
            model (torch.nn.Module): The model to test.
            data_loader (torch_geometric.data.DataLoader): The data loader for the test set.

        Returns:
            pd.DataFrame: The evaluation results.
        """
        return super().evaluate(model, data_loader)
