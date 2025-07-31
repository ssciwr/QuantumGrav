import torch
from typing import Callable, Any
import torch_geometric
from numpy import mean, std
from collections.abc import Iterable


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

    def evaluate(
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
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

    def report(self, losses: Iterable[Any]) -> None:
        """Report the evaluation results to stdout"""
        avg = mean(losses)
        sigma = std(losses)
        print(f"Average loss: {avg}, Standard deviation: {sigma}")
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
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
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
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
    ):
        """Validate the model on the given data loader.

        Args:
            model (torch.nn.Module): Model to validate.
            data_loader (torch_geometric.loader.DataLoader): Data loader for validation.
        Returns:
            list[Any]: A list of validation results.
        """
        return self.evaluate(model, data_loader)
