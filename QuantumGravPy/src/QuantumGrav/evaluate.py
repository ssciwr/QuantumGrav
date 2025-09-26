import torch
from typing import Callable, Any, Iterable, Sequence
import torch_geometric
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import logging
import tqdm


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


# early stopping class. this checks a validation metric and stops training if it doesn´t improve anymore
class PandasEarlyStopping(DefaultEarlyStopping):
    # put this into the package
    def __init__(
        self, patience: int, delta: float = 1e-4, window=7, metric: str = "loss"
    ):
        super().__init__(
            patience=patience,
            delta=delta,
            window=window,
        )
        self.metric = metric
        self.best_score = -np.inf  # we want to maximize the metric

    def __call__(self, data: Sequence | pd.DataFrame | pd.Series) -> bool:
        self.found_better = False

        if self.best_score + self.delta < data[self.metric].iloc[-1]:
            self.logger.info(
                f"    Better model found: {data[self.metric].iloc[-1]}, current best: {self.best_score}"
            )
            self.best_score = data[self.metric].iloc[-1]  # record best score
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

        return self.current_patience <= 0


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

                target.append(data.y.cpu())

                outputs = ((torch.sigmoid(predictions[0].squeeze()) > 0.5).long()).cpu()
                output.append(outputs)

                if loss.isnan().any():
                    print(f"NaN loss encountered in batch {i}.")
                    continue

                if np.isnan(output).any():
                    print(f"NaN encountered in output in batch {i}.")
                    continue

                if torch.isnan(data.y).any():
                    print(f"NaN target encountered in batch {i}.")
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

    def report(self, data: pd.DataFrame | dict):
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
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        super().__init__(device, criterion, apply_model, prefix="Validation")

    def validate(self, model, data_loader):
        return super().evaluate(model, data_loader)


class F1Tester(F1Evaluator):
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        super().__init__(device, criterion, apply_model, prefix="Test")

    def test(self, model, data_loader):
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

                target.append(data.y.cpu())

                outputs = ((torch.sigmoid(predictions[0].squeeze()) > 0.5).long()).cpu()
                output.append(outputs)

                if loss.isnan().any():
                    print(f"NaN loss encountered in batch {i}.")
                    continue

                if np.isnan(output).any():
                    print(f"NaN encountered in output in batch {i}.")
                    continue

                if torch.isnan(data.y).any():
                    print(f"NaN target encountered in batch {i}.")
                    continue

        current_data = pd.DataFrame(
            {
                "loss": torch.cat(losses),
                "output": torch.cat(output),
                "target": torch.cat(target),
            }
        )
        avg_loss = current_data["loss"].mean()
        std_loss = current_data["loss"].std()
        accuracy = (
            (current_data["output"] == current_data["target"]).float().mean().item()
        )
        mse = torch.nn.functional.mse_loss(
            current_data["output"].float(), current_data["target"].float()
        ).item()
        mae = torch.nn.functional.l1_loss(
            current_data["output"].float(), current_data["target"].float()
        ).item()

        self.data.loc[len(self.data)] = [avg_loss, std_loss, accuracy, mse, mae]

        return current_data

    def report(self, data: pd.DataFrame | dict):
        """Report the evaluation results.

        Args:
            data (pd.DataFrame | dict): The evaluation data.
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
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        super().__init__(device, criterion, apply_model, prefix="Validation")

    def validate(self, model, data_loader):
        return super().evaluate(model, data_loader)


class AccuracyTester(AccuracyEvaluator):
    def __init__(
        self,
        device,
        criterion,
        apply_model=None,
    ):
        super().__init__(device, criterion, apply_model, prefix="Test")

    def test(self, model, data_loader):
        return super().evaluate(model, data_loader)
