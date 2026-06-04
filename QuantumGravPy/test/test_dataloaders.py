import logging
from copy import deepcopy

import jsonschema
import pytest
import torch

from QuantumGrav import dataloaders


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size=10, label="dataset", **kwargs):
        self.size = size
        self.label = label
        self.kwargs = kwargs
        self.selected_indices = None
        self.shuffle_calls = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"label": self.label, "index": index}

    def index_select(self, indices):
        selected = FakeDataset(
            len(indices), label=f"{self.label}-subset", **self.kwargs
        )
        selected.selected_indices = list(indices)
        return selected

    def shuffle(self):
        self.shuffle_calls += 1
        return self


class RecordingDataLoader:
    calls = []

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.sampler = kwargs.get("sampler")
        RecordingDataLoader.calls.append(self)

    def __len__(self):
        return len(self.dataset)


class FakeDistributedSampler:
    calls = []

    def __init__(self, dataset, num_replicas, rank, shuffle, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epochs = []
        FakeDistributedSampler.calls.append(self)

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


@pytest.fixture(autouse=True)
def reset_recorders():
    RecordingDataLoader.calls = []
    FakeDistributedSampler.calls = []


@pytest.fixture
def base_config():
    return {
        "name": "dataloader-test",
        "log_level": "DEBUG",
        "training": {
            "seed": 123,
            "batch_size": 4,
            "num_workers": 1,
            "pin_memory": False,
            "drop_last": True,
            "prefetch_factor": None,
            "persistent_workers": True,
            "shuffle": True,
        },
        "validation": {
            "batch_size": 5,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "prefetch_factor": None,
            "shuffle": False,
        },
        "testing": {
            "batch_size": 6,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "prefetch_factor": None,
            "shuffle": False,
        },
    }


@pytest.fixture
def data_config(tmp_path):
    return {
        "files": ["input-a.zarr", "input-b.zarr"],
        "output": str(tmp_path / "processed"),
        "validate_data": False,
        "float_type": torch.float64,
        "int_type": torch.int64,
        "chunksize": 3,
        "n_processes": 0,
        "transform": lambda data: data,
        "pre_transform": lambda data: data,
        "pre_filter": lambda data: True,
    }


def make_factory(config):
    return dataloaders.DataLoaderFactory(
        config, logger=logging.getLogger("test-dataloaders")
    )


def test_init_validates_schema_sets_logger_level_and_seed(base_config):
    factory = make_factory(base_config)

    assert factory.config is base_config
    assert factory.logger.level == logging.DEBUG
    assert factory.nprng.integers(0, 100, size=5).tolist() == [1, 68, 59, 5, 90]
    assert factory.torch_generator.initial_seed() == 123

    invalid = deepcopy(base_config)
    invalid["training"].pop("seed")
    with pytest.raises(jsonschema.ValidationError):
        dataloaders.DataLoaderFactory(invalid)


def test_from_config_uses_module_logger(base_config):
    factory = dataloaders.DataLoaderFactory.from_config(base_config)

    assert factory.logger.name == dataloaders.__name__
    assert factory.logger.level == logging.DEBUG


@pytest.mark.parametrize(
    ("section", "updates"),
    [
        ("training", {"num_workers": 0, "persistent_workers": True}),
        ("validation", {"num_workers": 0, "persistent_workers": True}),
        ("testing", {"num_workers": 0, "persistent_workers": True}),
        ("training", {"num_workers": 0, "prefetch_factor": 2}),
        ("validation", {"num_workers": 0, "prefetch_factor": 2}),
        ("testing", {"num_workers": 0, "prefetch_factor": 2}),
        ("training", {"persistent_workers": True, "num_workers": None}),
        ("validation", {"prefetch_factor": 2, "num_workers": None}),
    ],
)
def test_schema_rejects_worker_options_that_require_workers(
    base_config, section, updates
):
    invalid = deepcopy(base_config)
    invalid[section].update(updates)
    if updates.get("num_workers") is None:
        invalid[section].pop("num_workers")

    with pytest.raises(jsonschema.ValidationError):
        dataloaders.DataLoaderFactory(invalid)


def test_schema_accepts_worker_options_with_positive_workers(base_config):
    valid = deepcopy(base_config)
    valid["training"].update(num_workers=1, persistent_workers=True, prefetch_factor=2)
    valid["validation"].update(
        num_workers=2, persistent_workers=True, prefetch_factor=3
    )
    valid["testing"].update(num_workers=1, persistent_workers=True, prefetch_factor=2)

    dataloaders.DataLoaderFactory(valid)


def test_build_dataset_from_config_passes_options_applies_subset_and_shuffle(
    base_config, data_config, monkeypatch
):
    created = []

    class CapturingDataset(FakeDataset):
        def __init__(self, **kwargs):
            super().__init__(size=10, label="built", **kwargs)
            created.append(self)

    monkeypatch.setattr(dataloaders.dataset_ondisk, "QGDataset", CapturingDataset)
    data_config = {**data_config, "subset": 0.25, "shuffle": True}
    factory = make_factory(base_config)

    dataset = factory._build_dataset_from_config(data_config, "data")

    assert created[0].kwargs == {
        "input": data_config["files"],
        "output": data_config["output"],
        "float_type": torch.float64,
        "int_type": torch.int64,
        "validate_data": False,
        "chunksize": 3,
        "n_processes": 0,
        "transform": data_config["transform"],
        "pre_transform": data_config["pre_transform"],
        "pre_filter": data_config["pre_filter"],
    }
    assert len(dataset) == 3
    assert dataset.selected_indices == [5, 6, 0]
    assert dataset.shuffle_calls == 1


def test_build_dataset_requires_data_config(base_config):
    factory = make_factory(base_config)

    with pytest.raises(
        ValueError, match="A 'training' data config section is required"
    ):
        factory._build_dataset_from_config(None, "training")


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (
            lambda cfg, dc: (
                cfg["training"].update(data=dc),
                cfg["validation"].update(data=dc),
                cfg["testing"].update(data=dc),
            ),
            "stage_local",
        ),
        (
            lambda cfg, dc: (cfg.update(data=dc), cfg["testing"].update(data=dc)),
            "shared_train_validation",
        ),
        (lambda cfg, dc: cfg.update(data=dc), "top_level_full"),
    ],
)
def test_resolve_data_mode_supported_shapes(
    base_config, data_config, mutator, expected
):
    mutator(base_config, data_config)
    assert make_factory(base_config)._resolve_data_mode() == expected


def test_resolve_data_mode_rejects_missing_or_mixed_data(base_config, data_config):
    with pytest.raises(ValueError, match="A 'data' config section is required"):
        make_factory(base_config)._resolve_data_mode()

    mixed = deepcopy(base_config)
    mixed["data"] = data_config
    mixed["training"]["data"] = data_config
    with pytest.raises(ValueError, match="Unsupported data config"):
        make_factory(mixed)._resolve_data_mode()


def test_validate_split_checks_length_and_sum(base_config):
    factory = make_factory(base_config)

    factory._validate_split([0.6, 0.3, 0.1], expected_parts=3)
    with pytest.raises(ValueError, match="must contain 3 values"):
        factory._validate_split([0.5, 0.5], expected_parts=3)
    with pytest.raises(ValueError, match="Splits must sum to one"):
        factory._validate_split([0.5, 0.4, 0.2], expected_parts=3)


def test_split_dataset_three_way_uses_documented_rounding_and_preserves_items(
    base_config,
):
    dataset = FakeDataset(size=11)

    train, val, test = make_factory(base_config)._split_dataset(
        dataset, [0.5, 0.25, 0.25]
    )

    assert [len(train), len(val), len(test)] == [6, 2, 3]
    assert train.dataset is dataset
    assert val.dataset is dataset
    assert test.dataset is dataset


def test_split_dataset_uses_factory_torch_generator_not_global_rng(base_config):
    dataset = FakeDataset(size=12)

    torch.manual_seed(999)
    _ = torch.rand(25)
    first_split = make_factory(deepcopy(base_config))._split_dataset(
        dataset, [0.5, 0.25, 0.25]
    )

    torch.manual_seed(111)
    _ = torch.rand(7)
    second_split = make_factory(deepcopy(base_config))._split_dataset(
        dataset, [0.5, 0.25, 0.25]
    )

    assert [subset.indices for subset in first_split] == [
        subset.indices for subset in second_split
    ]


def test_split_dataset_two_way_uses_factory_torch_generator(base_config):
    dataset = FakeDataset(size=10)

    first_train, first_val = make_factory(deepcopy(base_config))._split_dataset(
        dataset, [0.6, 0.4]
    )
    second_train, second_val = make_factory(deepcopy(base_config))._split_dataset(
        dataset, [0.6, 0.4]
    )

    assert first_train.indices == second_train.indices
    assert first_val.indices == second_val.indices


@pytest.mark.parametrize(
    ("size", "split", "message"),
    [
        (0, [0.8, 0.1, 0.1], "train size cannot be 0"),
        (3, [1.0, 0.0, 0.0], "validation size cannot be 0"),
        (3, [0.5, 0.5, 0.0], "test size cannot be 0"),
        (0, [0.8, 0.2], "train size cannot be 0"),
        (3, [1.0, 0.0], "validation size cannot be 0"),
    ],
)
def test_split_dataset_rejects_empty_splits(base_config, size, split, message):
    with pytest.raises(ValueError, match=message):
        make_factory(base_config)._split_dataset(FakeDataset(size=size), split)


def test_prepare_datasets_stage_local_builds_each_stage(
    base_config, data_config, monkeypatch
):
    built = []

    def fake_build(data, stage_name):
        built.append((stage_name, data))
        return FakeDataset(
            size={"training": 3, "validation": 4, "testing": 5}[stage_name]
        )

    for section in ("training", "validation", "testing"):
        base_config[section]["data"] = {
            **data_config,
            "output": f"{data_config['output']}-{section}",
        }
    factory = make_factory(base_config)
    monkeypatch.setattr(factory, "_build_dataset_from_config", fake_build)

    train, val, test = factory._prepare_datasets_from_config([0.8, 0.1, 0.1])

    assert [len(train), len(val), len(test)] == [3, 4, 5]
    assert [stage for stage, _ in built] == ["training", "validation", "testing"]


def test_prepare_datasets_shared_train_validation_splits_shared_data_and_builds_test(
    base_config, data_config, monkeypatch
):
    built = []

    def fake_build(data, stage_name):
        built.append(stage_name)
        return FakeDataset(size=10 if stage_name == "data" else 4, label=stage_name)

    base_config["data"] = {**data_config, "split": [0.6, 0.4]}
    base_config["testing"]["data"] = {
        **data_config,
        "output": f"{data_config['output']}-test",
    }
    factory = make_factory(base_config)
    monkeypatch.setattr(factory, "_build_dataset_from_config", fake_build)

    train, val, test = factory._prepare_datasets_from_config([0.8, 0.2])

    assert [len(train), len(val), len(test)] == [6, 4, 4]
    assert train.dataset is val.dataset
    assert built == ["data", "testing"]


def test_prepare_datasets_top_level_full_uses_three_way_split(
    base_config, data_config, monkeypatch
):
    base_config["data"] = {**data_config, "split": [0.5, 0.25, 0.25]}
    factory = make_factory(base_config)
    monkeypatch.setattr(
        factory,
        "_build_dataset_from_config",
        lambda data, stage_name: FakeDataset(size=8, label=stage_name),
    )

    train, val, test = factory._prepare_datasets_from_config([0.8, 0.1, 0.1])

    assert [len(train), len(val), len(test)] == [4, 2, 2]


def test_prepare_dataset_rejects_full_dataset_with_explicit_splits(base_config):
    factory = make_factory(base_config)

    with pytest.raises(ValueError, match="full dataset must not be provided"):
        factory.prepare_dataset(dataset=FakeDataset(), train_dataset=FakeDataset())


def test_prepare_dataset_splits_supplied_dataset_or_returns_explicit_datasets(
    base_config, monkeypatch
):
    factory = make_factory(base_config)
    train, val, test = factory.prepare_dataset(
        dataset=FakeDataset(size=10), split=[0.6, 0.2, 0.2]
    )
    assert [len(train), len(val), len(test)] == [6, 2, 2]

    explicit = (FakeDataset(1, "train"), FakeDataset(2, "val"), FakeDataset(3, "test"))
    factory = make_factory(base_config)
    monkeypatch.setattr(
        factory,
        "_build_dataset_from_config",
        lambda data, stage_name: pytest.fail("explicit datasets should not build data"),
    )
    assert (
        factory.prepare_dataset(
            train_dataset=explicit[0], val_dataset=explicit[1], test_dataset=explicit[2]
        )
        == explicit
    )


def test_loader_kwargs_reads_stage_local_persistent_workers(base_config):
    factory = make_factory(base_config)

    assert factory._loader_kwargs("training")["persistent_workers"] is True
    assert factory._loader_kwargs("validation")["persistent_workers"] is False

    base_config["validation"].update(num_workers=1, persistent_workers=True)
    kwargs = make_factory(base_config)._loader_kwargs("validation")

    assert kwargs == {
        "batch_size": 5,
        "num_workers": 1,
        "pin_memory": True,
        "drop_last": False,
        "prefetch_factor": None,
        "persistent_workers": True,
        "shuffle": False,
    }


def test_prepare_dataloaders_builds_loaders_and_sampler_disables_train_shuffle(
    base_config, monkeypatch
):
    monkeypatch.setattr(dataloaders, "DataLoader", RecordingDataLoader)
    sampler = object()
    factory = make_factory(base_config)
    monkeypatch.setattr(
        factory,
        "_build_dataset_from_config",
        lambda data, stage_name: pytest.fail("explicit datasets should not build data"),
    )

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        train_dataset=FakeDataset(3),
        val_dataset=FakeDataset(2),
        test_dataset=FakeDataset(1),
        training_sampler=sampler,
    )

    assert (train_loader, val_loader, test_loader) == tuple(RecordingDataLoader.calls)
    assert train_loader.kwargs["sampler"] is sampler
    assert train_loader.kwargs["shuffle"] is False
    assert val_loader.kwargs["shuffle"] is False
    assert test_loader.kwargs["batch_size"] == 6


def test_distributed_factory_uses_config_rank_or_override(base_config):
    config = {**base_config, "parallel": {"world_size": 4, "rank": 2}}

    factory = dataloaders.DistributedDataLoaderFactory(config)

    assert factory.rank == 2
    assert dataloaders.DistributedDataLoaderFactory(config, rank=1).rank == 1
    assert factory.world_size == 4
    assert factory.sampler_seed == base_config["training"]["seed"]


def test_distributed_prepare_dataloaders_creates_partition_samplers_and_disables_shuffle(
    base_config, monkeypatch
):
    monkeypatch.setattr(dataloaders, "DataLoader", RecordingDataLoader)
    monkeypatch.setattr(torch.utils.data, "DistributedSampler", FakeDistributedSampler)
    config = {**base_config, "parallel": {"world_size": 3, "rank": 1}}
    factory = dataloaders.DistributedDataLoaderFactory(config)
    monkeypatch.setattr(
        factory,
        "_build_dataset_from_config",
        lambda data, stage_name: pytest.fail("explicit datasets should not build data"),
    )

    train_loader, val_loader, test_loader = factory.prepare_dataloaders(
        train_dataset=FakeDataset(9),
        val_dataset=FakeDataset(6),
        test_dataset=FakeDataset(3),
    )

    assert (train_loader, val_loader, test_loader) == tuple(RecordingDataLoader.calls)
    assert [sampler.shuffle for sampler in FakeDistributedSampler.calls] == [
        True,
        False,
        False,
    ]
    assert [sampler.num_replicas for sampler in FakeDistributedSampler.calls] == [
        3,
        3,
        3,
    ]
    assert [sampler.rank for sampler in FakeDistributedSampler.calls] == [1, 1, 1]
    assert [sampler.seed for sampler in FakeDistributedSampler.calls] == [123, 123, 123]
    assert train_loader.kwargs["shuffle"] is False
    assert val_loader.kwargs["shuffle"] is False
    assert test_loader.kwargs["shuffle"] is False
    assert factory.train_sampler is FakeDistributedSampler.calls[0]
    assert factory.val_sampler is FakeDistributedSampler.calls[1]
    assert factory.test_sampler is FakeDistributedSampler.calls[2]

    factory.set_epoch(7)
    assert [sampler.epochs for sampler in FakeDistributedSampler.calls] == [
        [7],
        [7],
        [7],
    ]


def test_distributed_set_epoch_requires_prepared_samplers(base_config):
    config = {**base_config, "parallel": {"world_size": 2, "rank": 0}}
    factory = dataloaders.DistributedDataLoaderFactory(config)

    with pytest.raises(
        RuntimeError, match="Distributed samplers have not been prepared"
    ):
        factory.set_epoch(0)


def test_distributed_prepare_dataloaders_honors_training_sampler_override(
    base_config, monkeypatch
):
    monkeypatch.setattr(dataloaders, "DataLoader", RecordingDataLoader)
    monkeypatch.setattr(torch.utils.data, "DistributedSampler", FakeDistributedSampler)
    config = {**base_config, "parallel": {"world_size": 2, "rank": 0}}
    factory = dataloaders.DistributedDataLoaderFactory(config)
    monkeypatch.setattr(
        factory,
        "_build_dataset_from_config",
        lambda data, stage_name: pytest.fail("explicit datasets should not build data"),
    )
    custom_sampler = object()

    train_loader, _, _ = factory.prepare_dataloaders(
        train_dataset=FakeDataset(4),
        val_dataset=FakeDataset(4),
        test_dataset=FakeDataset(4),
        training_sampler=custom_sampler,
    )

    assert factory.train_sampler is custom_sampler
    assert train_loader.kwargs["sampler"] is custom_sampler
    assert [sampler.shuffle for sampler in FakeDistributedSampler.calls] == [
        False,
        False,
    ]
