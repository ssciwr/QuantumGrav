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

- A `pre_transform` function which builds the actual `Data` object. This will be executed only once when you open the dataset path.  Internally, the dataset will create a directory named `processed` which will contain the processed files, one for each cset. The precence of this directory is used to determine if `pre_transform` is executed again, so you can go to the directory and delete `processed` or rename it to trigger a new processing run. The `pre_transform` function is optional, it defaults to `None` and you don't have to provide it. This can be useful when you experiment a lot with different compositions, subsets or transformations. The `pre_transform` function is optional, it defaults to `None` and you don't have to provide it. This can be useful when you experiment a lot with different compositions, subsets or transformations.
If not supplied, the dataset will return the return value of the `reader` function for each loaded datapoint.
- A `pre_filter` function which filters out undesired raw samples and only lets a subset through to be processed by `pre_transform`. The semantics is the same as `pre_transform`, and the two will always be executed together. Like the `pre_transform` function, the `pre_filter` function is optional. Not supplying it will make the dataset use every datapoint regardless of properties. `pre_filter` and `pre_transform` function independently, only when both are `None` is the cached pre-processing not triggered.

- A `transform` function which is executed each time a datapoint is loaded from disk. This works well when doing experimentations with different data transformations or for augmentation on the fly.

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
def reader(store, idx, float_dtype, int_dtype, validate):
    group = zarr.open_group(store.root)
    adj = group["adjacency_matrix"][idx, :, :]
    cset_size = int(group["cset_size"][idx])
    data = {
        "adjacency_matrix": adj,
        "cset_size": cset_size,
        "feature_names": [
            "max_pathlens_future",
            "max_pathlens_past",
            "in_degree",
            "out_degree",
        ],
        "manifold_like": int(group["manifold_like"][idx]),
    }
    for name in data["feature_names"]:
        vals = torch.tensor(group[name][idx, :][:cset_size], dtype=float_dtype).unsqueeze(1)
        data[name] = torch.nan_to_num(vals, nan=0.0)
    return data
```

Now we have a function that turns raw data into a dictionary. Next, we need the `pre_filter` and `pre_transform` functions. We want to retain all data, so `pre_filter` can just return true all the time:

```python
pre_filter = lambda x: True
```
or we can filter out some targets:

```python
pre_filter = lambda data: data.y[2] != 2
```

Then, we need the `pre_transform` function which truns the returned dict into a `torch_geometric.data.Data` object. Here, we want to fix the adjacency matrix because Julia uses a different convention. we could have done this right away in the reader function, too, but it's a good way to show what `pre_transform` can do.

```python
def pre_transform(data: dict) -> Data:
    adj = np.transpose(data["adjacency_matrix"])  # if stored transposed
    adj = adj[: data["cset_size"], : data["cset_size"]]
    edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))
    x = torch.cat([data[name] for name in data["feature_names"]], dim=1).to(torch.float32)
    y = torch.tensor(data["manifold_like"], dtype=torch.long)
    tg = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    if not tg.validate():
        raise ValueError("Data validation failed.")
    return tg
```

The `transform` function follows the same principle, so we don't show it explicitly here and just set it to a no-op:
```python
transform = lambda x: x
```
Note how the `reader` function returns a dictionary, while the `pre_transform` function builds the data object that `pytorch_geometric` works with. If `pre_transform` is not given, a dataset would load datapoints as `dicts`.


Now, we can put together our dataset. Upon first instantiation, it will pre-process the data in the files given as `input` into `Data` objects and store them individually in a directory `output/processed`. As long as it sees this directory, it will not process any files again when another dataset is opend with the same output path. Data procesessing will be parallelized over the number of processes given as `n_processes`, and `chunk_size` many samples will be processed at once.

```python
dataset = QGDataset(
    input=input,
    output=output,
    reader=reader,
    validate_data=True,
    chunksize=1000,
    n_processes=4,
    transform=transform,
    pre_transform=pre_transform,
    pre_filter=pre_filter,
)
```
Processing runs in `chunksize` batches using `n_processes` workers.


## Indexing and subsetting
To index into an existing dataset, use the normal `[]` operator with either a range or an integer:
```python
datapoint = dataset[42]

range_of_datapoints = dataset[42:84]
```

Note that the dataset internally treats all supplied files as one consecutive range of datapoints, in the order they are given!

Since the `QGDataset` is a torch dataset under the hood, it works together with [torch's subset functionality](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset):

```python
subset = torch.utils.data.Subset(dataset, list(range(0, 100, 4)))
random_subset = torch.utils.data.Subset(dataset, torch.randint(0, len(dataset), (25,)).tolist())
```

This can help with dataset splitting for, e.g., k-fold cross validation or other tasks where splitting a dataset into multiple groups are needed.