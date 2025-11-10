# Using Datasets for data processing and batching

## Raw data
`QuantumGrav` supports Zarr as a data format for storing the raw data. Each cset, by default, is stored in it's own group in the file. In most cases, you want one index, typically the first or the last, to be used as the sample index, such that each array has $$N = n_{sample} + 1$$ dimensions, with $$n_{sample}$$ being the dimensionality of the data for a single sample.
For practicality, this should be the same in each stored array.

## Concept
The various Dataset all work by the same principle:

- read in all data that is needed to construct a complete `pytorch_geometric.data.Data` object, which represents a single cset graph, its feature data and its target data. The former will be processed by the graph neural network models this package is designed to build, the latter are the targets for supervised learning tasks.
- Build the features (node-level, graph-level or edge-level) and targets from the raw data.
- Make a `Data` object and save it to disk via `torch.save`.

Becausse this package does not make assumptions about the structure and character of the input raw data, you need to supply a set of functions yourself:

- A `reader` function that reads the raw data to construct a single cset/graph/sample.

- A `pre_transform` function which builds the actual `Data` object. This will be executed only once when you open the dataset path.  Internally, the dataset will create a directory named `processed` which will contain the processed files, one for each cset. The precence of this directory is used to determine if `pre_transform` is executed again, so you can go to the directory and delete `processed` or rename it to trigger a new processing run.

- A `pre_filter` function which filters out undesired raw samples and only lets a subset through to be processed by `pre_transform`. The semantics is the same as `pre_transform`, and the two will always be executed together.

- A `transform` function which is executed each time the dataset path on disk is opened, and can be used to execute all data transformations that would need to be carried out each time a dataset is loaded.

The last three are part of `pytorch`/`pytorch_geometric`'s `Dataset` API, so check out the respective documentation to learn more about them.

## How to create a dataset object
Each dataset must first know where the raw data is stored. A set of .zarr files can be passed as a list.

Next, it needs to know where to store the processed data. This is given by a single `pathlib.Path` or `string` object. A `processed` directory will be created there and the result of `pretransform` for each sample will be stored there.
Let's inspect the signature of the constructor:

```python
class QGDataset(QGDatasetBase, Dataset):
    def __init__(
        self,
        input: list[str | Path],
        output: str | Path,
        reader: Callable[[zarr.Group, torch.dtype, torch.dtype, bool], list[Data]] | None = None,
        float_type: torch.dtype = torch.float32,
        int_type: torch.dtype = torch.int64,
        validate_data: bool = True,
        chunksize: int = 1000,
        n_processes: int = 1,
        # dataset properties
        transform: Callable[[Data | Collection], Data] | None = None,
        pre_transform: Callable[[Data | Collection], Data] | None = None,
        pre_filter: Callable[[Data | Collection], bool] | None = None,
    )
```

First, we need to define the list of input files. We also need to choose the output directory.  If we used `zarr` files, we would put `zarr` here.

```python
  input = ['path/to/file1.zarr', 'path/to/file2.zarr', 'path/to/file3.zarr']
  output = 'path/to/output'
```

Then, we have to define the function that reads data from the file. This eats a zarr file, a float and int type, and whether to validate the data or not. For example:

```python
def reader(
        f: zarr.Group, idx: int, float_dtype: torch.dtype, int_dtype: torch.dtype, validate: bool
    ) -> Data:

        # get the adjacency matrix
        adj_raw = f["adjacency_matrix"][idx, :, :]
        adj_matrix = torch.tensor(adj_raw, dtype=float_dtype)
        node_features = []

        # Path lengths
        max_path_future = torch.tensor(
            f["max_pathlen_future"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor

        max_path_past = torch.tensor(
            f["max_pathlen_past"][idx, :], dtype=float_dtype
        ).unsqueeze(1)  # make this a (num_nodes, 1) tensor
        node_features.extend([max_path_future, max_path_past])

        # make the targets
        manifold = f["manifold"][idx]
        boundary = f["boundary"][idx]
        dimension = f["dimension"][idx]

        if (
            isinstance(manifold, np.ndarray)
            and isinstance(boundary, np.ndarray)
            and isinstance(dimension, np.ndarray)
        ):
            value_list = [manifold.item(), boundary.item(), dimension.item()]
        else:
            value_list = [manifold, boundary, dimension]

        return {
            "adj": adj_matrix,
            "max_pathlen_future": max_path_future,
            "max_pathlen_past": max_path_past,
            "manifold": manifold,
            "boundary": boundary,
            "dimension": dimension
        }
```

Now we have a function that turns raw data into a dictionary. Next, we need the `pre_filter` and `pre_transform` functions. We want to retain all data, so `pre_filter` can just return true all the time:

```python
pre_filter = lambda x: true
```
or we can filter out some targets:

```python
pre_filter = lambda data: data.y[2] != 2
```

Then, we need the `pre_transform` function which truns the returned dict into a `torch_geometric.data.Data` object. Here, we want to fix the adjacency matrix because Julia uses a different convention. we could have done this right away in the reader function, too, but it's a good way to show what `pre_transform` can do.

```python
def pre_transform(data: Data) -> Data:
    """Pre-transform the data dictionary into a  Data object."""
    adjacency_matrix = data["adjacency_matrix"]
    cset_size = data["cset_size"]
    # this is a workaround for the fact that the adjacency matrix is stored in a transposed form when going from julia to hdf5
    adjacency_matrix = np.transpose(adjacency_matrix)
    adjacency_matrix = adjacency_matrix[0:cset_size, 0:cset_size]
    edge_index, edge_weight = dense_to_sparse(
        torch.tensor(adjacency_matrix, dtype=torch.float32)
    )

    node_features = []
    for feature_name in data["feature_names"]:
        node_features.append(data[feature_name])
    x = torch.cat(node_features, dim=1).to(torch.float32)
    y = torch.tensor(data["manifold_like"]).to(torch.long)

    # make data object
    tgdata = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y,
    )

    if not tgdata.validate():
        raise ValueError(f"Data validation failed for index {idx}.")
    return tgdata
```

The `transform` function follows the same principle, so we don't show it explicitly here and just set it to a no-op:
```python
transform = lambda x: x
```
Now, we can put together our dataset. Upon first instantiation, it will pre-process the data in the files given as `input` into `Data` objects and store them individually in a directory `output/processed`. As long as it sees this directory, it will not process any files again when another dataset is opend with the same output path. Data procesessing will be parallelized over the number of processes given as `n_processes`, and `chunk_size` many samples will be processed at once.

```python
dataset = QGDataset(
    input,
    output,
    reader = reader,
    validate_data = True,
    chunksize = 5000,
    n_processes= 12,
    transform = transform,
    pre_transform = pre_transform,
    pre_filter = pre_filter,
)
```
Here we use 12 processes which process the data in chunks of 5000 samples before loading the next 5000 using the `reader` function, processing them and so on.

## OntheFly dataset
We will rarely use this, so no explicit example is provided. You can check out the `test_ontheflydataset.py` test file to see how it is used in principle.