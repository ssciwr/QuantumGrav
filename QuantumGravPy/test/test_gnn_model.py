import QuantumGrav as QG
import pytest

import torch
import torch_geometric

from functools import partial


@pytest.fixture
def encoder_type():
    """Fixture providing an encoder class type"""
    return QG.models.GNNBlock


@pytest.fixture
def encoder_args():
    """Fixture providing encoder constructor args"""
    return [16, 32]


@pytest.fixture
def encoder_kwargs():
    """Fixture providing encoder constructor kwargs"""
    return {
        "dropout": 0.3,
        "gnn_layer_type": torch_geometric.nn.conv.GCNConv,
        "normalizer_type": torch.nn.BatchNorm1d,
        "activation_type": torch.nn.ReLU,
        "gnn_layer_args": [],
        "gnn_layer_kwargs": {"cached": False, "bias": True, "add_self_loops": True},
        "norm_args": [32],
        "norm_kwargs": {"eps": 1e-5, "momentum": 0.2},
        "skip_args": [16, 32],
        "skip_kwargs": {"weight_initializer": "kaiming_uniform"},
    }


@pytest.fixture
def gnn_block(encoder_type, encoder_args, encoder_kwargs):
    return encoder_type(*encoder_args, **encoder_kwargs)


@pytest.fixture
def downstream_task_specs():
    """Fixture providing downstream task specifications"""
    return [
        (
            QG.models.LinearSequential,
            [
                [(64, 32), (32, 12), (12, 2)],
                [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
            ],
            {
                "linear_kwargs": [{"bias": True}, {"bias": True}, {"bias": False}],
                "activation_kwargs": [{"inplace": False}, {}, {}],
            },
        ),
        (
            QG.models.LinearSequential,
            [
                [(64, 18), (18, 3)],
                [torch.nn.ReLU, torch.nn.Identity],
            ],
            {
                "linear_kwargs": [{"bias": True}, {"bias": False}],
                "activation_kwargs": [{}, {}],
            },
        ),
    ]


@pytest.fixture
def pooling_specs():
    """Fixture providing pooling layer specifications"""
    return [
        [torch_geometric.nn.global_mean_pool, [], {}],
        [torch_geometric.nn.global_max_pool, [], {}],
    ]


@pytest.fixture
def downstream_tasks(downstream_task_specs):
    first_type, first_args, first_kwargs = downstream_task_specs[0]
    second_type, second_args, second_kwargs = downstream_task_specs[1]
    return [
        (first_type(*first_args, **first_kwargs), None, None),
        (second_type(*second_args, **second_kwargs), None, None),
    ]


@pytest.fixture
def downstream_tasks_graphfeatures():
    specs = [
        [
            QG.models.LinearSequential,
            [
                [(96, 32), (32, 12), (12, 2)],
                [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
            ],
            {
                "linear_kwargs": [{"bias": True}, {"bias": True}, {"bias": False}],
                "activation_kwargs": [{"inplace": False}, {}, {}],
            },
        ],
        [
            QG.models.LinearSequential,
            [
                [(96, 18), (18, 3)],
                [torch.nn.ReLU, torch.nn.Identity],
            ],
            {
                "linear_kwargs": [{"bias": True}, {"bias": False}],
                "activation_kwargs": [{}, {}],
            },
        ],
    ]

    return specs


@pytest.fixture
def pooling_layer(pooling_specs):
    return pooling_specs


@pytest.fixture
def latent_model():
    nn = QG.models.LinearSequential(
        [
            (32, 24),
            (24, 64),
        ],
        [torch.nn.ReLU, torch.nn.Identity],
        linear_kwargs=[{"bias": True}, {"bias": False}],
        activation_kwargs=[{"inplace": False}, {"inplace": False}],
    )
    return nn


@pytest.fixture
def gnn_model(gnn_block, downstream_tasks, pooling_layer):
    return QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=downstream_tasks,
        pooling_layers=pooling_layer,
        aggregate_pooling_type=partial(torch.cat, dim=1),
        aggregate_pooling_args=None,
        aggregate_pooling_kwargs=None,
        active_tasks={0: True, 1: True},
    )


# Helper functions for aggregation
def concat(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, y), dim=1)


@pytest.fixture
def gnn_model_with_graph_features(
    gnn_block, downstream_tasks_graphfeatures, pooling_layer
):
    graph_features_net = QG.models.LinearSequential(
        [(10, 32), (32, 24), (24, 32)],
        [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        linear_kwargs=[{"bias": True}, {"bias": True}, {"bias": False}],
        activation_kwargs=[{"inplace": False}, {"inplace": False}, {"inplace": False}],
    )
    return QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=downstream_tasks_graphfeatures,
        pooling_layers=pooling_layer,
        aggregate_graph_features_type=concat,
        graph_features_net_type=graph_features_net,
        aggregate_pooling_type=partial(torch.cat, dim=1),
        active_tasks={0: True, 1: True},
    )


@pytest.fixture
def gnn_model_config(encoder_type, encoder_args, encoder_kwargs, pooling_specs):
    """Config matching the new GNNModel.from_config API, using existing fixtures."""
    config = {
        "encoder_type": encoder_type,
        "encoder_args": encoder_args,
        "encoder_kwargs": encoder_kwargs,
        # After two pooling ops concatenated along dim=1, encoder output 32 -> 64 input to heads
        "downstream_tasks": [
            [
                QG.models.LinearSequential,
                [
                    [(96, 24), (24, 18), (18, 2)],
                    [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
                ],
                {
                    "linear_kwargs": [
                        {"bias": True},
                        {"bias": True},
                        {"bias": False},
                    ],
                    "activation_kwargs": [{"inplace": False}, {}, {}],
                },
            ],
            [
                QG.models.LinearSequential,
                [
                    [(96, 24), (24, 18), (18, 3)],
                    [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
                ],
                {
                    "linear_kwargs": [
                        {"bias": True},
                        {"bias": True},
                        {"bias": False},
                    ],
                    "activation_kwargs": [{"inplace": False}, {}, {}],
                },
            ],
        ],
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": partial(torch.cat, dim=1),
        "graph_features_net_type": QG.models.LinearSequential,
        "graph_features_net_args": [
            [(10, 32), (32, 24), (24, 32)],
            [torch.nn.ReLU, torch.nn.ReLU, torch.nn.Identity],
        ],
        "graph_features_net_kwargs": {
            "linear_kwargs": [{"bias": True}, {"bias": True}, {"bias": False}],
            "activation_kwargs": [
                {"inplace": False},
                {"inplace": False},
                {"inplace": False},
            ],
        },
        "aggregate_graph_features_type": concat,
        "active_tasks": {0: False, 1: True},
    }
    return config


@pytest.fixture
def arbitrary_model_cfg(pooling_specs):
    config = {
        "encoder_type": torch_geometric.nn.models.GCN,
        "encoder_args": [16, 32, 3],
        "encoder_kwargs": {
            "dropout": 0.5,
            "act": "relu",
            "act_first": False,
            "act_kwargs": None,
            "norm": "batchnorm",
            "norm_kwargs": None,
            "jk": None,
        },
        "downstream_tasks": [
            [
                torch_geometric.nn.models.MLP,
                [],
                {
                    "in_channels": 80,
                    "hidden_channels": 32,
                    "out_channels": 4,
                    "num_layers": 3,
                },
            ],
            [
                torch_geometric.nn.models.MLP,
                [],
                {
                    "in_channels": 80,
                    "hidden_channels": 24,
                    "out_channels": 4,
                    "num_layers": 4,
                },
            ],
            [
                torch_geometric.nn.models.MLP,
                [],
                {
                    "in_channels": 80,
                    "hidden_channels": 23,
                    "out_channels": 2,
                    "num_layers": 4,
                },
            ],
        ],
        "pooling_layers": pooling_specs,
        "aggregate_pooling_type": partial(torch.cat, dim=1),
        "graph_features_net_type": torch_geometric.nn.models.MLP,
        "graph_features_net_args": [],
        "graph_features_net_kwargs": {
            "in_channels": 8,
            "hidden_channels": 16,
            "out_channels": 16,
            "num_layers": 3,
        },
        "aggregate_graph_features_type": concat,
        "active_tasks": {
            0: True,
            1: False,
            2: True,
        },
    }

    return config


def test_module_wrapper_behavior():
    """Ensure ModuleWrapper forwards and exposes function."""

    def fn(x, y):
        return x + y

    wrapper = QG.gnn_model.ModuleWrapper(fn)
    out = wrapper(torch.tensor(1), torch.tensor(2))
    assert out.item() == 3
    assert wrapper.get_fn() is fn


def test_instantiate_type_variants():
    """Cover instantiate_type for Module, class, and callable."""
    mod = torch.nn.Linear(2, 2)
    got_mod = QG.gnn_model.instantiate_type(mod, None, None)
    assert isinstance(got_mod, torch.nn.Module)

    cls_inst = QG.gnn_model.instantiate_type(torch.nn.ReLU, None, {"inplace": False})
    assert isinstance(cls_inst, torch.nn.Module)

    def my_pool(a, b=None):
        return a if b is None else a

    callable_wrapped = QG.gnn_model.instantiate_type(my_pool, None, None)
    assert isinstance(callable_wrapped, QG.gnn_model.ModuleWrapper)


def test_gnn_model_creation_pooling(gnn_model):
    """Test the creation of GNNModel from pre-existing module instances."""

    # Test encoder is correctly set
    assert isinstance(gnn_model.encoder, QG.models.GNNBlock)
    assert gnn_model.encoder.in_dim == 16
    assert gnn_model.encoder.out_dim == 32

    # Test downstream tasks are correctly set
    assert isinstance(gnn_model.downstream_tasks, torch.nn.ModuleList)
    assert len(gnn_model.downstream_tasks) == 2
    assert all(
        isinstance(task, QG.models.LinearSequential)
        for task in gnn_model.downstream_tasks
    )

    # Test pooling layers are correctly set
    assert isinstance(gnn_model.pooling_layers, torch.nn.ModuleList)
    assert len(gnn_model.pooling_layers) == 2
    assert all(
        isinstance(layer, QG.gnn_model.ModuleWrapper)
        for layer in gnn_model.pooling_layers
    )

    # Test aggregate pooling is correctly set
    assert isinstance(gnn_model.aggregate_pooling, QG.gnn_model.ModuleWrapper)

    # Test active tasks
    assert gnn_model.active_tasks == {0: True, 1: True}

    # Test that all components are registered as modules
    module_names = {name for name, _ in gnn_model.named_modules()}
    assert "encoder" in module_names
    assert "downstream_tasks" in module_names
    assert "pooling_layers" in module_names

    # Test parameters exist and require gradients
    params = list(gnn_model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)

    # Test forward pass works
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    gnn_model.eval()
    output = gnn_model(x, edge_index, batch)

    assert isinstance(output, dict)
    assert 0 in output
    assert 1 in output
    assert output[0].shape[0] == 3  # 3 graphs in batch
    assert output[0].shape[1] == 2  # 3 output classes
    assert output[1].shape[0] == 3  # 3 graphs in batch
    assert output[1].shape[1] == 3  # 3 output classes


def test_gnn_model_creation_latent(
    gnn_block, downstream_tasks, pooling_layer, latent_model
):
    """Test the gnn model with a latent space model instead of a normal one"""

    gnn_model = QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=downstream_tasks,
        pooling_layers=None,
        aggregate_pooling_type=None,
        aggregate_pooling_args=None,
        aggregate_pooling_kwargs=None,
        latent_model_type=latent_model,
        latent_model_args=None,
        latent_model_kwargs=None,
        active_tasks={0: True, 1: True},
    )

    # Test encoder is correctly set
    assert isinstance(gnn_model.encoder, QG.models.GNNBlock)
    assert gnn_model.encoder.in_dim == 16
    assert gnn_model.encoder.out_dim == 32

    # Test downstream tasks are correctly set
    assert isinstance(gnn_model.downstream_tasks, torch.nn.ModuleList)
    assert len(gnn_model.downstream_tasks) == 2
    assert all(
        isinstance(task, QG.models.LinearSequential)
        for task in gnn_model.downstream_tasks
    )

    # Test that latent model is set correctly and that with_latent is correct
    assert gnn_model.with_latent is True
    assert gnn_model.with_pooling is False

    # Test that all components are registered as modules
    module_names = {name for name, _ in gnn_model.named_modules()}
    assert "encoder" in module_names
    assert "downstream_tasks" in module_names
    assert "latent_model" in module_names

    # Test parameters exist and require gradients
    params = list(gnn_model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)

    # Test active tasks
    assert gnn_model.active_tasks == {0: True, 1: True}


def test_gnn_model_latent_forward(
    gnn_block, downstream_tasks, pooling_layer, latent_model
):
    """Test the gnn model with a latent space model instead of a normal one"""

    gnn_model = QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=downstream_tasks,
        latent_model_type=latent_model,
        latent_model_args=None,
        latent_model_kwargs=None,
        active_tasks={0: True, 1: True},
    )

    # Test forward pass works
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    gnn_model.eval()
    output = gnn_model(x, edge_index, batch)

    assert isinstance(output, dict)
    assert 0 in output
    assert 1 in output
    assert output[0].shape[0] == 10  # 3 graphs in batch
    assert output[0].shape[1] == 2  # 3 output classes
    assert output[1].shape[0] == 10  # 3 graphs in batch
    assert output[1].shape[1] == 3  # 3 output classes


def test_gnn_model_creation_goes_wrong():
    """Test the creation of GNNModel with preexisting modules with errors"""

    # Test: empty downstream_tasks
    with pytest.raises(
        ValueError, match="At least one downstream task must be provided"
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[],
        )

    # Test: pooling_layers without aggregate_pooling_type
    with pytest.raises(
        ValueError,
        match="If pooling layers are to be used, both an aggregate pooling method and pooling layers must be provided.",
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            pooling_layers=[(torch_geometric.nn.global_mean_pool, None, None)],
            aggregate_pooling_type=None,
        )

    # Test: aggregate_pooling_type without pooling_layers
    with pytest.raises(
        ValueError,
        match="If pooling layers are to be used, both an aggregate pooling method and pooling layers must be provided.",
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            pooling_layers=None,
            aggregate_pooling_type=torch.cat,
        )

    # Test: graph_features_net_type without aggregate_graph_features_type
    with pytest.raises(
        ValueError,
        match="If graph features are to be used, both a graph features network and an aggregation method must be provided.",
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            graph_features_net_type=torch.nn.Linear(5, 10),
            aggregate_graph_features_type=None,
        )

    # Test: aggregate_graph_features_type without graph_features_net_type
    with pytest.raises(
        ValueError, match="both a graph features network and an aggregation method"
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            graph_features_net_type=None,
            aggregate_graph_features_type=torch.cat,
        )

    # Test: empty pooling_layers list
    with pytest.raises(ValueError, match="At least one pooling layer must be provided"):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            pooling_layers=[],
            aggregate_pooling_type=torch.cat,
        )

    with pytest.raises(
        ValueError,
        match="active_tasks keys must match the indices of downstream tasks.",
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            pooling_layers=[(torch_geometric.nn.global_mean_pool, None, None)],
            aggregate_pooling_type=torch.cat,
            active_tasks={
                5: True,
            },
        )

    with pytest.raises(
        ValueError,
        match="active_tasks keys must match the indices of downstream tasks.",
    ):
        QG.GNNModel(
            encoder_type=torch.nn.Identity(),
            downstream_tasks=[(torch.nn.Linear(10, 5), None, None)],
            pooling_layers=[(torch_geometric.nn.global_mean_pool, None, None)],
            aggregate_pooling_type=torch.cat,
            active_tasks={0: True, 1: True, 2: False},
        )


def test_gnn_model_creation_from_scratch(
    encoder_type, encoder_args, encoder_kwargs, downstream_task_specs, pooling_specs
):
    """Test the creation of GNNModel with all parts being constructed from scratch"""

    model = QG.GNNModel(
        encoder_type=encoder_type,
        encoder_args=encoder_args,
        encoder_kwargs=encoder_kwargs,
        downstream_tasks=downstream_task_specs,
        pooling_layers=pooling_specs,
        aggregate_pooling_type=partial(torch.cat, dim=1),
        active_tasks={0: True, 1: False},
    )

    # Test encoder was constructed correctly
    assert isinstance(model.encoder, QG.models.GNNBlock)
    assert model.encoder.in_dim == 16
    assert model.encoder.out_dim == 32
    assert model.encoder.dropout_p == 0.3

    # Test downstream tasks were constructed correctly
    assert len(model.downstream_tasks) == 2
    assert isinstance(model.downstream_tasks[0], QG.models.LinearSequential)
    assert isinstance(model.downstream_tasks[1], QG.models.LinearSequential)

    # Test pooling layers were constructed correctly
    assert len(model.pooling_layers) == 2
    assert all(
        isinstance(layer, QG.gnn_model.ModuleWrapper) for layer in model.pooling_layers
    )

    # Test aggregate pooling
    assert isinstance(model.aggregate_pooling, QG.gnn_model.ModuleWrapper)

    # Test active tasks
    assert model.active_tasks == {0: True, 1: False}

    # Test stored specs for reconstruction
    assert model.encoder_args == encoder_args
    assert model.encoder_kwargs == encoder_kwargs
    assert model.downstream_task_specs == downstream_task_specs
    assert model.pooling_layer_specs == pooling_specs
    assert isinstance(model.graph_features_net, torch.nn.Identity)
    assert model.graph_features_net_args is None
    assert model.graph_features_net_kwargs is None

    # Test forward pass works
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    model.eval()
    output = model(x, edge_index, batch)

    # Only task 0 should be active
    assert isinstance(output, dict)
    assert 0 in output
    assert 1 not in output
    assert output[0].shape[0] == 3  # 3 graphs in batch
    assert output[0].shape[1] == 2  # 3 output classes


def test_gnn_model_creation_from_scratch_goes_wrong(
    encoder_type, encoder_args, encoder_kwargs
):
    """Test the creation of GNNModel with all parts being constructed from scratch with errors"""

    # Test: Invalid encoder args (wrong number of positional args)
    with pytest.raises((TypeError, RuntimeError)):
        QG.GNNModel(
            encoder_type=encoder_type,
            encoder_args=[16],  # Missing out_dim
            encoder_kwargs=encoder_kwargs,
            downstream_tasks=[
                (
                    QG.models.LinearSequential,
                    None,
                    {
                        "dims": [(32, 3)],
                        "activations": [torch.nn.Identity],
                    },
                )
            ],
        )

    # Test: Invalid downstream task spec (wrong type)
    with pytest.raises((ValueError, RuntimeError)):
        QG.GNNModel(
            encoder_type=encoder_type,
            encoder_args=encoder_args,
            encoder_kwargs=encoder_kwargs,
            downstream_tasks=[("not_a_valid_type", None, {"dims": [(32, 3)]})],
        )

    # Test: Incompatible dimensions between encoder and downstream tasks
    model = QG.GNNModel(
        encoder_type=encoder_type,
        encoder_args=encoder_args,
        encoder_kwargs=encoder_kwargs,
        downstream_tasks=[
            (
                QG.models.LinearSequential,
                None,
                {
                    "dims": [(64, 3)],  # Expects 64 but encoder outputs 32
                    "activations": [torch.nn.Identity],
                },
            )
        ],
        pooling_layers=[(torch_geometric.nn.global_mean_pool, None, None)],
        aggregate_pooling_type=torch.cat,
    )

    # Should fail during forward pass due to dimension mismatch
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1])

    with pytest.raises(RuntimeError):
        model(x, edge_index, batch)


def test_gnn_model_get_embeddings(gnn_model):
    """Test the get_embeddings method of GNNModel."""

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model.get_embeddings(x, edge_index, batch)
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 64)  # concatenated over pooling layers

    output_single = gnn_model.get_embeddings(x, edge_index)  # no batch -> single graph
    assert isinstance(output_single, torch.Tensor)
    assert output_single.shape == (1, 64)  # concatenated over pooling layers


def test_gnn_model_forward(gnn_model):
    "test gnn model forward run"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    output = gnn_model(x, edge_index, batch)
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert isinstance(output, dict)
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs, 2 classes


def test_gnn_model_forward_set_active(gnn_model):
    "test the model forward pass with active/inactive tasks"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.eval()  # Set model to evaluation mode
    assert gnn_model.training is False  # Ensure model is in eval mode
    assert gnn_model.active_tasks == {0: True, 1: True}
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 2
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs,  3 classes

    gnn_model.set_task_inactive(1)
    assert gnn_model.active_tasks == {0: True, 1: False}
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 1
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes

    gnn_model.set_task_active(1)
    assert gnn_model.active_tasks == {0: True, 1: True}
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 2
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs,  3 classes

    gnn_model.set_task_inactive(0)
    assert gnn_model.active_tasks == {0: False, 1: True}
    output = gnn_model(x, edge_index, batch)
    assert len(output) == 1
    assert output[1].shape == (2, 3)  # 2 graphs,  3 classes


# parameterized to make sure all paths through the computations graph can be backward traced
@pytest.mark.parametrize("to_disable", [1, 0])
def test_gnn_model_set_active_backprop(gnn_model, to_disable):
    "test the model forward pass with active/inactive tasks and backpropagation"
    task_to_keep = 0 if to_disable == 1 else 1

    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    gnn_model.train()  # Set model to training mode

    y = [torch.randn(2, 2), torch.randn(2, 3)]  # Random target values
    compute_loss = torch.nn.MSELoss()

    # baseline - both tasks active
    assert gnn_model.training is True  # Ensure model is in train mode
    assert gnn_model.active_tasks == {0: True, 1: True}
    output = gnn_model(x, edge_index, batch)
    assert 0 in output and 1 in output
    loss_terms = [
        compute_loss(output[i], y[i])
        for i in range(len(gnn_model.active_tasks))
        if gnn_model.active_tasks[i]
    ]
    loss = torch.stack(loss_terms).sum()

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.named_parameters():
        assert p.grad is not None

    assert len(output) == 2
    assert output[0].shape == (2, 2)  # 2 graphs, 2 concat pooling layers, 3 classes
    assert output[1].shape == (2, 3)  # 2 graphs, 2 concat pooling layers, 2 classes

    # reset grads
    for p in gnn_model.parameters():
        p.grad = None

    # disable task 'to_disable'
    gnn_model.set_task_inactive(to_disable)
    assert gnn_model.active_tasks[to_disable] is False
    assert gnn_model.active_tasks[task_to_keep] is True
    output = gnn_model(x, edge_index, batch)
    loss_terms = [
        compute_loss(output[i], y[i])
        for i in range(len(gnn_model.active_tasks))
        if gnn_model.active_tasks[i]
    ]
    loss = torch.stack(loss_terms).sum()

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.encoder.named_parameters():
        assert p.grad is not None

    for _, p in gnn_model.downstream_tasks[task_to_keep].named_parameters():
        assert p.grad is not None

    for _, p in gnn_model.downstream_tasks[to_disable].named_parameters():
        assert p.grad is None

    assert len(output) == 1
    assert output[task_to_keep].shape == (
        2,
        2 if to_disable == 1 else 3,
    )  # 2 graphs, 2 classes
    # reset grads
    for p in gnn_model.parameters():
        p.grad = None

    # re-enable task 'to_disable'
    gnn_model.set_task_active(to_disable)
    assert gnn_model.active_tasks == {0: True, 1: True}
    output = gnn_model(x, edge_index, batch)
    loss_terms = [
        compute_loss(output[i], y[i])
        for i in range(len(gnn_model.active_tasks))
        if gnn_model.active_tasks[i]
    ]
    loss = torch.stack(loss_terms).sum()

    for _, p in gnn_model.named_parameters():
        assert p.grad is None

    loss.backward()

    for _, p in gnn_model.named_parameters():
        assert p.grad is not None

    assert len(output) == 2
    assert output[0].shape == (2, 2)  # 2 graphs, 2 classes
    assert output[1].shape == (2, 3)  # 2 graphs, 3 classes
    # reset grads
    for p in gnn_model.parameters():
        p.grad = None


def test_gnn_model_forward_with_graph_features(gnn_model_with_graph_features):
    "test gnn model forward run with graph features"
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 10)  # 2 graphs with 10 features each
    gnn_model_with_graph_features.eval()  # Set model to evaluation mode
    output = gnn_model_with_graph_features(
        x, edge_index, batch, graph_features=graph_features
    )
    assert (
        gnn_model_with_graph_features.training is False
    )  # Ensure model is in eval mode
    assert isinstance(output, dict)
    assert output[0].shape == (2, 2)  # 2 graphs, 1 pooling layers, 2 classes
    assert output[1].shape == (2, 3)


def test_gnn_model_creation_from_config(gnn_model_config):
    "test gnn model initialization from config file"
    model = QG.GNNModel.from_config(gnn_model_config)

    # Active tasks mapping
    assert model.active_tasks == {0: False, 1: True}

    # Encoder is a single module
    assert isinstance(model.encoder, QG.models.GNNBlock)

    # Downstream tasks are built
    assert isinstance(model.downstream_tasks, torch.nn.ModuleList)
    assert len(model.downstream_tasks) == 2
    assert all(
        isinstance(t, QG.models.LinearSequential) for t in model.downstream_tasks
    )

    # Pooling layers exist and are wrappers around functions
    assert model.pooling_layers is not None
    for pooling in model.pooling_layers:
        assert isinstance(pooling, QG.gnn_model.ModuleWrapper)

    # Graph features network provided in config
    assert isinstance(model.graph_features_net, QG.models.LinearSequential)

    # Run a forward pass
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1])
    model.eval()
    output = model(x, edge_index, batch, graph_features=torch.randn(2, 10))
    assert model.training is False
    assert isinstance(output, dict)
    assert 0 not in output  # inactive
    assert 1 in output
    assert output[1].shape == (2, 3)


def test_gnn_model_save_load(gnn_model_config, tmp_path):
    "test saving and loading of the combined model"
    og_model = QG.GNNModel.from_config(gnn_model_config)
    og_model.save(tmp_path / "model.pt")
    assert (tmp_path / "model.pt").exists()

    loaded_gnn_model = QG.GNNModel.load(tmp_path / "model.pt", gnn_model_config)
    assert loaded_gnn_model.graph_features_net is not None
    assert len(loaded_gnn_model.state_dict().keys()) == len(
        og_model.state_dict().keys()
    )
    for k in loaded_gnn_model.state_dict().keys():
        assert torch.equal(
            loaded_gnn_model.state_dict()[k],
            og_model.state_dict()[k],
        )

    loaded_keys = set(loaded_gnn_model.state_dict().keys())
    original_keys = set(og_model.state_dict().keys())
    assert loaded_keys == original_keys

    for k in loaded_gnn_model.state_dict().keys():
        assert torch.equal(
            loaded_gnn_model.state_dict()[k],
            og_model.state_dict()[k],
        )

    og_model.eval()
    loaded_gnn_model.eval()
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 10)  # 2 graphs with 10 features each

    y = og_model.forward(x, edge_index, batch, graph_features=graph_features)

    y_loaded = loaded_gnn_model.forward(
        x, edge_index, batch, graph_features=graph_features
    )
    for i in y.keys():
        assert torch.allclose(y[i], y_loaded[i], atol=1e-8)


def test_gnn_model_arbitrary_composition(arbitrary_model_cfg, tmp_path):
    "test saving and loading of the combined model with arbitrary composition"

    gnn_model = QG.GNNModel.from_config(arbitrary_model_cfg)
    assert isinstance(gnn_model.encoder, torch_geometric.nn.models.GCN)
    assert len(gnn_model.downstream_tasks) == 3
    for task in gnn_model.downstream_tasks:
        assert isinstance(task, torch_geometric.nn.models.MLP)
    assert isinstance(gnn_model.graph_features_net, torch_geometric.nn.models.MLP)

    gnn_model.save(tmp_path / "arbitrary_model.pt")
    assert (tmp_path / "arbitrary_model.pt").exists()
    loaded_gnn_model = QG.GNNModel.load(
        tmp_path / "arbitrary_model.pt",
        arbitrary_model_cfg,
    )
    assert len(loaded_gnn_model.state_dict().keys()) == len(
        gnn_model.state_dict().keys()
    )
    for k in loaded_gnn_model.state_dict().keys():
        assert torch.equal(
            loaded_gnn_model.state_dict()[k],
            gnn_model.state_dict()[k],
        )

    # test forward pass
    x = torch.randn(5, 16)  # 5 nodes with 16 features each
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )  # Simple edge index
    batch = torch.tensor([0, 0, 0, 1, 1])  # Two graphs in the batch
    graph_features = torch.randn(2, 8)  # 2 graphs with 8 features each

    gnn_model.eval()
    loaded_gnn_model.eval()
    y = gnn_model.forward(x, edge_index, batch, graph_features=graph_features)
    y_loaded = loaded_gnn_model.forward(
        x, edge_index, batch, graph_features=graph_features
    )
    for i in y.keys():
        assert torch.allclose(y[i], y_loaded[i], atol=1e-8)


def test_gnn_model_compute_downstream_args_kwargs(gnn_block, pooling_layer):
    """Ensure downstream args/kwargs are passed into tasks."""
    head = QG.models.LinearSequential(
        [(64, 16), (16, 2)],
        [torch.nn.ReLU, torch.nn.Identity],
        linear_kwargs=[{"bias": True}, {"bias": False}],
        activation_kwargs=[{"inplace": False}, {"inplace": False}],
    )
    model = QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=[(head, None, None), (head, None, None)],
        pooling_layers=pooling_layer,
        aggregate_pooling_type=partial(torch.cat, dim=1),
        active_tasks={0: True, 1: True},
    )

    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    model.eval()
    outputs = model(
        x,
        edge_index,
        batch,
        downstream_task_args=[[], []],
        downstream_task_kwargs=[{}, {}],
    )
    assert 0 in outputs and 1 in outputs
    assert outputs[0].shape == (2, 2)


def test_set_task_active_inactive_errors(gnn_model):
    with pytest.raises(KeyError):
        gnn_model.set_task_active(5)
    with pytest.raises(KeyError):
        gnn_model.set_task_inactive(5)


def test_get_embeddings_no_pooling_path(gnn_block):
    """Exercise path when with_pooling is False and latent is used."""
    latent = QG.models.LinearSequential(
        [(32, 24)],
        [torch.nn.Identity],
        linear_kwargs=[{"bias": True}],
        activation_kwargs=[{}],
    )
    model = QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=[(torch.nn.Identity(), None, None)],
        latent_model_type=latent,
        active_tasks={0: True},
    )
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    out = model.get_embeddings(x, edge_index)
    assert isinstance(out, torch.Tensor)
    assert out.shape[-1] == 24


def test_graph_features_paths(gnn_block, pooling_layer):
    """Cover graph feature aggregation identity fallback when config not provided."""
    model = QG.GNNModel(
        encoder_type=gnn_block,
        downstream_tasks=[
            (QG.models.LinearSequential([(64, 2)], [torch.nn.Identity]), None, None)
        ],
        pooling_layers=pooling_layer,
        aggregate_pooling_type=partial(torch.cat, dim=1),
        active_tasks={0: True},
    )
    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    # since graph_features_net and aggregation are Identity, do not pass features
    model.eval()
    outputs = model(x, edge_index, batch)
    assert 0 in outputs


def test_from_config_validation_error():
    bad_cfg = {"encoder_type": torch.nn.Identity}
    with pytest.raises(RuntimeError):
        QG.GNNModel.from_config(bad_cfg)
