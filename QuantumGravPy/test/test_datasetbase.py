import pytest
import QuantumGrav as QG
from pathlib import Path
import torch
from torch_geometric.data import Data
import h5py


def test_dataset_base_creation(create_data, tmp_path):
    _, datafiles = create_data

    dataset = QG.dataset_base.QGDatasetBase(
        input=datafiles,
        output=tmp_path,
        reader=lambda file: [],
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


def test_dataset_base_creation_fails_bad_datafile(create_data, tmp_path):
    _, datafiles = create_data
    with pytest.raises(
        FileNotFoundError, match="Input file nonexistent_path does not exist."
    ):
        QG.dataset_base.QGDatasetBase(
            input=datafiles
            + [
                "nonexistent_path",
            ],
            output=tmp_path,
            reader=lambda file: [],
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


def test_dataset_base_creation_fails_no_data_reader(create_data, tmp_path):
    _, datafiles = create_data
    with pytest.raises(
        ValueError, match="A reader function must be provided to load the data."
    ):
        QG.dataset_base.QGDatasetBase(
            input=datafiles,
            output=tmp_path,
            reader=None,
            float_type=torch.float32,
            int_type=torch.int64,
            validate_data=True,
        )


@pytest.mark.parametrize("n", [1, 2], ids=["sequential", "parallel"])
def test_dataset_base_process_chunk_sequential(create_data, read_data, n):
    datadir, datafiles = create_data

    dataset = QG.dataset_base.QGDatasetBase(
        input=datafiles,
        output=datadir,
        reader=read_data,
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        n_processes=n,
        chunksize=4,
    )
    with h5py.File(datafiles[0], "r") as raw_file:
        results = dataset.process_chunk(
            raw_file,
            0,
            pre_transform=lambda x: x,
            pre_filter=lambda x: True,
        )

    assert len(results) == 4
    assert all(isinstance(res, Data) for res in results)
