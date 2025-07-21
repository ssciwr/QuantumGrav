import QuantumGrav as QG
import torch


def test_ondisk_dataset_creation_works(create_data, tmp_path):
    datadir, datafiles = create_data

    dataset = QG.QGDataset(
        input=datafiles,
        output=tmp_path,
        get_metadata=lambda x: {"num_samples": len(x)},
        reader=lambda file: [],
        float_type=torch.float32,
        int_type=torch.int64,
        validate_data=True,
        chunksize=1100,
        n_processes=3,
        transform=lambda x: x,
        pre_transform=lambda x: x,
        pre_filter=lambda x: True,
    )

    assert dataset.input == datafiles
    assert dataset.output == tmp_path
    assert isinstance(dataset.get_metadata, callable)
    assert dataset.float_type == torch.float32
    assert dataset.int_type == torch.int64
    assert dataset.validate_data is True
    assert dataset.reader is not None
    assert dataset.chunksize == 1100
    assert dataset.n_processes == 3


def test_ondisk_dataset_process_data():
    assert 3 == 6


def test_ondisk_dataset_creation_fails_no_reader():
    assert 3 == 6


def test_ondisk_dataset_processing(create_data):
    assert 3 == 6
