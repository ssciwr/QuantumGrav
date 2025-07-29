from typing import Callable, Any
import torch
import torch_geometric
from collections.abc import Collection


from . import ddp
from . import gnn_model


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
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Evaluate a single batch of data using the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch_geometric.data.Data): The input data for the model.
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): A function to apply the model to the data. Defaults to None.

    Returns:
        torch.Tensor | tuple[torch.Tensor, ...]: The output of the model.
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
    criterion: Callable[
        [torch.Tensor, torch.Tensor | torch_geometric.data.Data], torch.Tensor | float
    ],
    loss_data: list[dict[str, Any]],
    evaluate_result: Callable[
        [
            list[dict[str, Any]],
            torch.Tensor | float,
            Collection[float] | Collection[torch.Tensor],
        ],
        list[dict[str, Any]],
    ] = make_loss_statistics,
    report_result: Callable[[list[dict[str, Any]]], None] = print,
    device: torch.device = torch.device("cpu"),
    apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any] = None,
    parallel: bool = False,
    rank: int = 0,  # Only used if parallel is True
) -> None:
    """Train the model for one epoch. This will put the model into training mode, iterate over the data loader, compute the loss, and update the model parameters. It also processes the loss data if a processing function is provided or computes statistics on the loss data if not. Either output will be appended to the loss_data list.

    Args:
        model (torch.nn.Module): Model to train
        data_loader (torch_geometric.data.DataLoader): Data loader for the training data
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor  |  float]): Loss function to compute the loss
        loss_data (list[dict[str, Any]]): list to store loss statistics
        evaluate_result (Callable[ [torch.Tensor  |  float, Collection[float], Collection[torch.Tensor]], list[dict[str, Any]], ], optional): Function to process the loss data. Defaults to None.
        device (torch.device, optional): Device to run the training on. Defaults to torch.device("cpu").
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): Function to apply the model to the data. Defaults to None.
        parallel (bool, optional): Whether to use distributed training. Defaults to False.
        rank (int, optional): Rank of the current process in distributed training. Defaults to 0. Only used if parallel is True.
    """
    if torch.distributed.is_initialized() is False and parallel:
        raise RuntimeError(
            "Distributed training is enabled, but torch.distributed is not initialized. "
            "Make sure to initialize the distributed environment before training."
        )

    if not model.training:
        raise RuntimeError(
            "Model should be in training mode before training. Use model.train() to set it to training mode."
        )

    # either both of evaluate_result and report_result must be provided or none
    if any(x is None for x in [evaluate_result, report_result]) and any(
        x is not None for x in [evaluate_result, report_result]
    ):
        raise ValueError(
            "evaluate_result and report_result must be provided for training."
        )

    epoch_loss_data = []

    # actual training loop
    for batch in data_loader:
        optimizer.zero_grad()
        data = batch.to(rank if parallel else device, non_blocking=True)
        outputs = evaluate_batch(model, data, apply_model)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        if isinstance(loss, torch.Tensor):
            epoch_loss_data.append(loss.item())
        else:
            epoch_loss_data.append(loss)

    if evaluate_result is not None and epoch_loss_data is not None:
        if parallel and rank == 0:
            torch.distributed.barrier()  # Ensure all processes reach this point before proceeding

            result = evaluate_result(
                loss_data, epoch_loss_data, data_loader.dataset.data.y
            )
            report_result(result)
            torch.distributed.barrier()  # Ensure all processes complete before returning
        else:
            result = evaluate_result(
                loss_data, epoch_loss_data, data_loader.dataset.data.y
            )
            report_result(result)


def test_epoch(
    model: torch.nn.Module,
    data_loader: torch_geometric.data.DataLoader,
    criterion: Callable[
        [torch.Tensor, torch.Tensor | torch_geometric.data.Data], torch.Tensor | float
    ],
    loss_data: list[dict[str, Any]],
    evaluate_result: Callable[
        [
            list[dict[str, Any]],
            torch.Tensor | float,
            Collection[float] | Collection[torch.Tensor],
        ],
        list[dict[str, Any]],
    ] = make_loss_statistics,
    report_result: Callable[[list[dict[str, Any]]], None] = print,
    device: torch.device = torch.device("cpu"),
    apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any] = None,
    parallel: bool = False,
    rank: int = 0,  # Only used if parallel is True
) -> None:
    """Test the model for one epoch.

    Args:
        model (torch.nn.Module): The model to test.
        data_loader (torch_geometric.data.DataLoader): The data loader for the test data.
        criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor  |  float]): The loss function.
        loss_data (list[dict[str, Any]]): list to store loss statistics.
        evaluate_result (Callable[ [torch.Tensor  |  float, Collection[float], Collection[torch.Tensor]], list[dict[str, Any]], ], optional): Function to process the loss data. Defaults to make_loss_statistics.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device("cpu").
        apply_model (Callable[[torch.nn.Module, torch_geometric.data.Data], Any], optional): A function to apply the model to the data. Defaults to None.
        parallel (bool, optional): Whether to use distributed evaluation. Defaults to False.
        rank (int, optional): Rank of the current process in distributed evaluation. Defaults to 0. Only used if parallel is True.
    """
    if torch.distributed.is_initialized() is False and parallel:
        raise RuntimeError(
            "Distributed training is enabled, but torch.distributed is not initialized. "
            "Make sure to initialize the distributed environment before training."
        )

    def run_on_data(
        model: torch.nn.Module,
        data_loader: torch_geometric.data.DataLoader,
        criterion: Callable[
            [torch.Tensor, torch.Tensor | torch_geometric.data.Data],
            torch.Tensor | float,
        ],
        device: torch.device,
        evaluate_result: Callable[
            [
                list[dict[str, Any]],
                torch.Tensor | float,
                Collection[float] | Collection[torch.Tensor],
            ],
            list[dict[str, Any]],
        ],
        report_result: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        """Evaluate the model on a given data loader. This function abstracts away the logic of evaluating the model on a data loader between parallel and non-parallel execution.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch_geometric.data.DataLoader): The data loader for the evaluation data.
            criterion (Callable[ [torch.Tensor, torch.Tensor  |  torch_geometric.data.Data], torch.Tensor  |  float, ]): The loss function.
            device (torch.device): The device to run the evaluation on.
            evaluate_result (Callable[ [ list[dict[str, Any]], torch.Tensor  |  float, Collection[float]  |  Collection[torch.Tensor], ], list[dict[str, Any]], ]): A function to process the evaluation results.
            report_result (Callable[[list[dict[str, Any]]], None]): A function to report the evaluation results.
        """
        model.eval()
        epoch_loss_data = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch.to(device, non_blocking=True)
                outputs = evaluate_batch(model, data, apply_model)
                loss = criterion(outputs, data)
                if isinstance(loss, torch.Tensor):
                    epoch_loss_data.append(loss.item())
                else:
                    epoch_loss_data.append(loss)

        if evaluate_result is not None and epoch_loss_data is not None:
            result = evaluate_result(
                loss_data, epoch_loss_data, data_loader.dataset.data.y
            )
            report_result(result)

    if parallel and rank == 0:
        torch.distributed.barrier()
        run_on_data(
            model, data_loader, criterion, device, evaluate_result, report_result
        )
        torch.distributed.barrier()
    else:
        run_on_data(
            model, data_loader, criterion, device, evaluate_result, report_result
        )


"""
Validate an epoch using the same logic as test_epoch.
"""
validate_epoch = test_epoch  # Re-use test_epoch logic for validation


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        # training and evaluation functions
        criterion: Callable[
            [torch.Tensor, torch.Tensor | torch_geometric.data.Data],
            torch.Tensor | float,
        ],
        apply_model: Callable[[torch.nn.Module, torch_geometric.data.Data], Any]
        | None = None,
        # training evaluation and reporting
        early_stopping: Callable[[list[dict[str, Any]]], bool] | None = None,
    ):
        self.config = config

        # functions for executing training and evaluation
        self.criterion = criterion
        self.apply_model = apply_model
        self.early_stopping = early_stopping
        self.seed = config.get("seed", 42)

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # parallel training
        self.parallel = config["training"].get("parallel", False)
        self.rank = config["parallel"].get("rank", 0)
        self.world_size = config["parallel"].get("world_size", 1)

        # early stopping parameters
        self.early_stopping_patience = config["training"].get(
            "early_stopping_patience", 10
        )
        self.early_stopping_counter = 0

        # parameters for finding out which model is best
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.epoch = 0
        self.checkpoint_at = config["training"].get("checkpoint_at", None)

        self.initialized_parallel = False

        self.model = None
        self.optimizer = None

    def _init_sequential(self):
        if self.model is not None:
            raise RuntimeError("Model is already initialized.")
        self.model = gnn_model.GNNModel.from_config(self.config["model"])

    def _init_parallel(self):
        if self.model is not None:
            raise RuntimeError("Model is already initialized.")

        if self.initialized_parallel:
            raise RuntimeError("Parallel training is already initialized.")

        if not self.parallel:
            raise RuntimeError("Parallel training is not enabled.")
        model = gnn_model.GNNModel.from_config(self.config["model"])

        ddp.initialize(
            rank=self.rank,
            worldsize=self.world_size,
            master_addr=self.config["parallel"].get("master_addr", "localhost"),
            master_port=self.config["parallel"].get("master_port", "12345"),
            backend=self.config["parallel"].get("backend", "nccl"),
        )

        self.model = ddp.DistributedModel(
            model,
            rank=self.rank,
            worldsize=self.world_size,
            output_device=self.config.get("output_device", None),
        )

        self.initialize_parallel = True

    def _init_optimizer(self):
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer.")

        if self.optimizer is not None:
            raise RuntimeError("Optimizer is already initialized.")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["training"].get("lr", 1e-3),
            weight_decay=self.config["training"].get("weight_decay", 0),
        )

    def run_training_sequential(
        self,
        train_loader: torch_geometric.data.DataLoader,
        val_loader: torch_geometric.data.DataLoader,
        evaluate_training: Callable[
            [
                list[dict[str, Any]],
                torch.Tensor | float,
                Collection[float] | Collection[torch.Tensor],
            ],
            list[dict[str, Any]],
        ] = make_loss_statistics,
        evaluate_validation: Callable[
            [
                list[dict[str, Any]],
                torch.Tensor | float,
                Collection[float] | Collection[torch.Tensor],
            ],
            list[dict[str, Any]],
        ] = make_loss_statistics,
        report_training_result=print,
        report_validation_result=print,
    ):
        self._init_sequential()
        self._init_optimizer()

        if self.model is None:
            raise RuntimeError("Model must be initialized before training.")

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before training.")

        num_epochs = self.config["training"].get("num_epochs", 100)
        self.model.train()
        total_eval_data = []
        for epoch in range(0, num_epochs):
            eval_data = []
            train_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.criterion,
                eval_data,
                device=self.config.get("device", torch.device("cpu")),
                apply_model=self.apply_model,
                evaluate_result=evaluate_training,
                report_result=report_training_result,
                parallel=False,
            )
            self.epoch += 1
            total_eval_data.append(eval_data)
            if self.checkpoint_at is not None and epoch % self.checkpoint_at == 0:
                self.save_checkpoint()

            if self.early_stopping is not None:
                if self.early_stopping(eval_data):
                    print(f"Early stopping at epoch {epoch}.")
                    self.save_checkpoint()
                    break

        self.model.eval()
        eval_val_data = []
        validate_epoch(
            self.model,
            val_loader,
            self.criterion,
            eval_val_data,
            evaluate_result=evaluate_validation,
            report_result=report_validation_result,
            device=self.config.get("device", torch.device("cpu")),
            apply_model=self.apply_model,
            parallel=False,
        )

    def run_training_ddp(
        self,
        train_loader: torch_geometric.data.DataLoader,
        val_loader: torch_geometric.data.DataLoader,
        evaluate_training: Callable[
            [
                list[dict[str, Any]],
                torch.Tensor | float,
                Collection[float] | Collection[torch.Tensor],
            ],
            list[dict[str, Any]],
        ] = make_loss_statistics,
        evaluate_validation: Callable[
            [
                list[dict[str, Any]],
                torch.Tensor | float,
                Collection[float] | Collection[torch.Tensor],
            ],
            list[dict[str, Any]],
        ] = make_loss_statistics,
        report_training_result=print,
        report_validation_result=print,
    ):
        self._init_parallel()
        self._init_optimizer()
        if self.model is None:
            raise RuntimeError("Model must be initialized before training.")

        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before training.")

        num_epochs = self.config["training"].get("num_epochs", 100)
        for epoch in range(0, num_epochs):
            eval_data = []
            train_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.criterion,
                eval_data,
                device=self.config.get("device", torch.device("cpu")),
                apply_model=self.apply_model,
                evaluate_result=evaluate_training,
                report_result=report_training_result,
                parallel=False,
            )
            self.epoch += 1

            torch.distributed.barrier()  # Ensure all processes reach this point before saving
            if self.checkpoint_at is not None and epoch % self.checkpoint_at == 0:
                if self.rank == 0:
                    self.save_checkpoint()

            torch.distributed.barrier()  # Ensure all processes complete the epoch before proceeding

            if self.rank == 0:
                self.model.eval()
                eval_val_data = []
                validate_epoch(
                    self.model,
                    val_loader,
                    self.criterion,
                    eval_val_data,
                    evaluate_result=evaluate_validation,
                    report_result=report_validation_result,
                    device=self.config.get("device", torch.device("cpu")),
                    apply_model=self.apply_model,
                    parallel=False,
                )
                torch.distributed.barrier()  # Ensure all processes complete the epoch before proceeding

            if self.early_stopping is not None:
                if self.early_stopping(eval_data):
                    print(f"Early stopping at epoch {epoch}.")
                    self.save_checkpoint()
                    break

            torch.distributed.barrier()

        self.cleanup()

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def run_test(self):
        if self.parallel:
            if self.rank == 0:
                pass
        else:
            pass

    def cleanup(self):
        if self.initialize_parallel:
            ddp.cleanup()
        else:
            pass
