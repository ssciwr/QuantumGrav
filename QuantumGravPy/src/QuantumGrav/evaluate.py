from typing import Callable, Any
import torch
import torch_geometric
from collections.abc import Collection
import tqdm


def make_loss_statistics(
    loss_data: list[dict[str, Any]],
    list_of_loss_values: list[Any],
    _: Collection[float] | None = None,
) -> None:
    """Generate loss statistics for a training/evaluation epoch.

    Args:
        loss_data (list[dict[str, Any]]): list to store loss statistics.
        list_of_loss_values (list[Any]): List of loss values for the epoch.
        y (Collection[float] | None, optional): Ground truth values. Defaults to None.

    Returns:
        list[dict[str, Any]]: Updated list with loss statistics.
    """

    # make statistics
    epoch_loss_data = torch.tensor(list_of_loss_values, dtype=torch.float32)
    mean_loss = torch.mean(epoch_loss_data).item()
    std_loss = torch.std(epoch_loss_data).item()
    min_loss = torch.min(epoch_loss_data).item()
    max_loss = torch.max(epoch_loss_data).item()
    median_loss = torch.median(epoch_loss_data).item()
    q25_loss = torch.quantile(epoch_loss_data, 0.25).item()
    q75_loss = torch.quantile(epoch_loss_data, 0.75).item()

    loss_data.append(
        {
            "mean": mean_loss,
            "std": std_loss,
            "min": min_loss,
            "max": max_loss,
            "median": median_loss,
            "q25": q25_loss,
            "q75": q75_loss,
        },
    )


def evaluate_batch(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any] = None,
) -> Any:
    """Evaluate a single batch of data using the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The input data for the model.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device("cpu").
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): A function to apply the model to the data. Defaults to None.

    Returns:
        Any: The output of the model.
    """
    if apply_model:
        outputs = apply_model(model, data)
    else:
        outputs = model(data.x, data.edge_index, data.batch)
    return outputs


def train_epoch(
    model: torch.nn.Module,
    data_loader: torch_geometric.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
    loss_data: list[dict[str, Any]],
    process_loss: Callable[
        [
            list[dict[str, Any]],
            torch.Tensor | float,
            Collection[float],
            Collection[torch.Tensor],
        ],
        list[dict[str, Any]],
    ] = make_loss_statistics,
    device: torch.device = torch.device("cpu"),
    apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any] = None,
) -> None:
    """Train the model for one epoch. This will put the model into training mode, iterate over the data loader, compute the loss, and update the model parameters. It also processes the loss data if a processing function is provided or computes statistics on the loss data if not. Either output will be appended to the loss_data list.

    Args:
        model (torch.nn.Module): Model to train
        data_loader (torch_geometric.data.DataLoader): Data loader for the training data
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor  |  float]): Loss function to compute the loss
        loss_data (list[dict[str, Any]]): list to store loss statistics
        process_loss (Callable[ [torch.Tensor  |  float, Collection[float], Collection[torch.Tensor]], list[dict[str, Any]], ], optional): Function to process the loss data. Defaults to None.
        device (torch.device, optional): Device to run the training on. Defaults to torch.device("cpu").
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): Function to apply the model to the data. Defaults to None.
    """
    if model.training is False:
        raise RuntimeError(
            "Model should be in training mode before training. Use model.train() to set it to training mode."
        )
    # actual training loop
    epoch_loss_data = []
    for batch in tqdm.tqdm(data_loader, desc="Training epoch"):
        optimizer.zero_grad()
        data = batch.to(device)
        outputs = evaluate_batch(model, data, apply_model)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        if isinstance(loss, torch.Tensor):
            epoch_loss_data.append(loss.item())
        else:
            epoch_loss_data.append(loss)

    process_loss(
        loss_data,
        epoch_loss_data,
        data_loader.dataset.data.y if hasattr(data_loader.dataset, "data") else None,
    )


def test_epoch(
    model: torch.nn.Module,
    data_loader: torch_geometric.data.DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
    loss_data: list[dict[str, Any]],
    process_loss: Callable[
        [torch.Tensor | float, Collection[float], Collection[torch.Tensor]],
        list[dict[str, Any]],
    ] = make_loss_statistics,
    device: torch.device = torch.device("cpu"),
    apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any] = None,
) -> None:
    """Test the model for one epoch.

    Args:
        model (torch.nn.Module): The model to test.
        data_loader (torch_geometric.data.DataLoader): The data loader for the test data.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor  |  float]): The loss function.
        loss_data (list[dict[str, Any]]): list to store loss statistics.
        process_loss (Callable[ [torch.Tensor  |  float, Collection[float], Collection[torch.Tensor]], list[dict[str, Any]], ], optional): Function to process the loss data. Defaults to make_loss_statistics.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device("cpu").
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): A function to apply the model to the data. Defaults to None.
    """
    model.eval()

    epoch_loss_data = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating epoch"):
            data = batch.to(device)
            outputs = evaluate_batch(model, data, apply_model)
            loss = criterion(outputs, data)
            if isinstance(loss, torch.Tensor):
                epoch_loss_data.append(loss.item())
            else:
                epoch_loss_data.append(loss)

    process_loss(
        loss_data,
        epoch_loss_data,
        data_loader.dataset.data.y if hasattr(data_loader.dataset, "data") else None,
    )


validate_epoch = test_epoch  # Re-use test_epoch logic for validation
