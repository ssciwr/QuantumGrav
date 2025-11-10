import pytest
from QuantumGrav.dataset_base import QGDatasetBase
from pathlib import Path
import torch
from torch_geometric.data import Data
import zarr


def test_dataset_base_creation(create_data_zarr, tmp_path):
    _, datafiles = create_data_zarr

    dataset = QGDatasetBase(
        input=datafiles,
        output=tmp_path,
        reader=lambda *args, **kwargs: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=4,
        chunksize=300,
    )

    assert dataset.input == datafiles
    assert dataset.output == tmp_path
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.data_reader is not None
    assert dataset.raw_file_names == [f.name for f in datafiles]
    assert dataset.chunksize == 300
    assert dataset.n_processes == 4
    assert Path(dataset.processed_dir).exists()
    assert (Path(dataset.processed_dir) / "metadata.yaml").exists()
    assert dataset.metadata["num_samples"] == 15
    assert dataset.metadata["input"] == [str(f.resolve().absolute()) for f in datafiles]
    assert dataset.metadata["output"] == str(Path(tmp_path).resolve().absolute())
    assert dataset.metadata["float_type"] == str(torch.float32)
    assert dataset.metadata["int_type"] == str(torch.int64)
    assert dataset.metadata["validate_data"] is True
    assert dataset.metadata["n_processes"] == 4
    assert dataset.metadata["chunksize"] == 300


def test_dataset_base_creation_fails_bad_datafile(create_data_zarr, tmp_path):
    _, datafiles = create_data_zarr

    with pytest.raises(
        FileNotFoundError, match="Input file nonexistent_path does not exist."
    ):
        QGDatasetBase(
            input=datafiles
            + [
                "nonexistent_path",
            ],
            output=tmp_path,
            reader=lambda *args, **kwargs: [],
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_base_creation_fails_no_data_reader(create_data_zarr, tmp_path):
    _, datafiles = create_data_zarr

    with pytest.raises(
        ValueError, match="A reader function must be provided to load the data."
    ):
        QGDatasetBase(
            input=datafiles,
            output=tmp_path,
            reader=None,
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_base_metadata_reload(create_data_zarr, tmp_path):
    # create initial dataset to write metadata
    _, datafiles = create_data_zarr
    ds1 = QGDatasetBase(
        input=datafiles,
        output=tmp_path,
        reader=lambda *args, **kwargs: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=1,
        chunksize=10,
    )
    meta_path = Path(ds1.processed_dir) / "metadata.yaml"
    assert meta_path.exists()

    # mutate metadata file externally to simulate prior run
    import yaml as _yaml

    with open(meta_path, "r") as f:
        meta = _yaml.load(f, Loader=_yaml.FullLoader)
    meta["chunksize"] = 999
    with open(meta_path, "w") as f:
        _yaml.dump(meta, f)

    # re-instantiate and ensure mutated value is read
    ds2 = QGDatasetBase(
        input=datafiles,
        output=tmp_path,
        reader=lambda *args, **kwargs: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=2,  # different but should be ignored in favor of file content
        chunksize=123,  # different but should be ignored in favor of file content
    )
    assert ds2.metadata["chunksize"] == 999


def test_dataset_base_processed_file_names_filter(tmp_path):
    # Prepare a fake processed directory structure
    processed_dir = Path(tmp_path) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    # valid files
    (processed_dir / "data_0.pt").write_text("x")
    (processed_dir / "data_other.pt").write_text("y")
    # invalid files (wrong extension or name)
    (processed_dir / "data.json").write_text("{}")
    (processed_dir / "random.pt").write_text("z")
    (processed_dir / "notdata.txt").write_text("w")
    # minimal metadata to allow instantiation logic to treat as existing processed dir
    import yaml as _yaml

    with open(processed_dir / "metadata.yaml", "w") as f:
        _yaml.dump({"num_samples": 0, "input": [], "output": str(tmp_path)}, f)

    ds = QGDatasetBase(
        input=[],
        output=tmp_path,
        reader=lambda *args, **kwargs: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=False,
    )
    assert sorted(ds.processed_file_names) == ["data_0.pt", "data_other.pt"]


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_dataset_base_process_chunk_filter_transform(create_data_zarr_basic, n):
    datadir, datafiles = create_data_zarr_basic

    def reader(group, index, ftype, itype, validate):
        # fabricate a minimal Data object encoding index so we can check transform/filter
        # we ignore the actual group contents for this synthetic test
        return Data(x=torch.tensor([[index]], dtype=ftype))  # type: ignore[return-value]

    ds = QGDatasetBase(  # type: ignore[arg-type]
        input=datafiles,
        output=datadir,
        reader=reader,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=6,
    )

    with zarr.storage.LocalStore(datafiles[0], read_only=True) as raw_file:  # type: ignore[attr-defined]
        # filter out even indices, and transform to add +100 to the feature
        results = ds.process_chunk(  # type: ignore[misc]
            raw_file,
            0,
            pre_transform=lambda d: Data(x=d.x + 100),  # type: ignore[misc]
            pre_filter=lambda d: int(d.x.item()) % 2 == 1,  # type: ignore[misc]
        )
    # Determine expected subset: indices 1,3,5 within first 6
    assert len(results) == 3
    xs = [int(r.x.item()) for r in results]  # type: ignore[misc]
    # after transform each original index should have +100 applied
    assert xs == [101, 103, 105]


@pytest.mark.parametrize("mode", ["fallback", "normal"], ids=["fallback", "normal"])
@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_dataset_base_process_chunk_zarr(
    read_data, create_data_zarr, create_data_zarr_basic, mode, n
):
    if mode == "fallback":
        datadir, datafiles = create_data_zarr_basic
    else:
        datadir, datafiles = create_data_zarr

    dataset = QGDatasetBase(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=4,
    )
    with zarr.storage.LocalStore(datafiles[0], read_only=True) as raw_file:  # type: ignore[attr-defined]
        results = dataset.process_chunk(  # type: ignore[misc]
            raw_file, 0, pre_transform=lambda x: x, pre_filter=lambda x: True
        )
    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)
