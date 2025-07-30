import torch
from typing import Callable
import torch_geometric
from numpy import mean, std


class DefaultEvaluator:
    def __init__(self, device, criterion: Callable, apply_model: Callable):
        self.criterion = criterion
        self.apply_model = apply_model
        self.device = device
        self.data = []

    def evaluate(
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
    ):
        model.eval()
        current_data = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                data = batch.to(self.device)
                if self.apply_model:
                    outputs = self.apply_model(data)
                else:
                    outputs = model(data)
                loss = self.criterion(outputs, data)
                current_data.append(loss)

        return current_data

    def report(self, losses):
        avg = mean(losses)
        sigma = std(losses)
        self.data.append((avg, sigma))


class DefaultTester(DefaultEvaluator):
    def __init__(self, device, criterion: Callable, apply_model: Callable):
        super().__init__(device, criterion, apply_model)

    def test(
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
    ):
        return self.evaluate(model, data_loader)


class DefaultValidator(DefaultEvaluator):
    def __init__(self, device, criterion: Callable, apply_model: Callable):
        super().__init__(device, criterion, apply_model)

    def validate(
        self, model: torch.nn.Module, data_loader: torch_geometric.loader.DataLoader
    ):
        return self.evaluate(model, data_loader)
