# Using Datasets for data processing and batching 

## Raw data 
`QuantumGrav` supports HDF5 and Zarr as raw data formats. Each of these can store n-dimensional arrays, in which our raw data will be stored. In most cases, you want one index, typically the first or the last, to be used as the sample index, such that each array has $$N = n_{sample} + 1$$ dimensions, with $$n_{sample}$$ being the dimensionality of the data for a single sample. 
For practicality, this should be the same in each stored array. 

## Concept 
The various Dataset all work by the same principle: 

- read in all data that is needed to construct a complete `pytorch_geometric.data.Data` object, which represents a single cset graph, its feature data and its target data. The former will be processed by the graph neural network models this package is designed to build, the latter are the targets for supervised learning tasks. 
- Build the features (node-level, graph-level or edge-level) and targets from the raw data. 
- Make a `Data` object and save it to disk via `torch.save`. 

Becausse this package does not make assumptions about the structure and character of the input raw data, you need to supply a set of functions yourself: 

- A `reader` function that reads the raw data to construct a single cset/graph/sample

- A `pretransform` function which builds the actual `Data` object. This will be executed only once and create a directory named `processed` which will contain the processed files, one for each cset. 

- A `prefilter` function which filters out undesired raw samples and only lets a subset through to be processed by `pretransform`. 

- A `transform` function which is executed each time the dataset path on disk is opened, and can be used to execute all data transformations that would need to be carried out each time a dataset is loaded. 

The last three are part of `pytorch`/`pytorch_geometric`'s `Dataset` API, so check out the respective documentation to learn more about them. 

## Examples for InMemory- and On-disk datasets
Each dataset must first know where the raw data is stored. This takes the form of one or more Zarr or HDF5 files, which are passed in as a list. 

Next, it needs to know where to store the processed data. This is given by a single `pathlib.Path` or `string` object. A `processed` directory will be created there and the result of `pretransform` for each sample will be stored there. 
While both `QGDatasetInMemory` and `QGDataset` store the processed data on disk, the former will load all the processed data into memory at once, while the latter will lazily load data when needed. 

So, let's start with the on-disk dataset `QGDataset`. The `InMemoryDataset` is treated in the same way. 


## OntheFly dataset 
We will rarely use this, so no explicit example is provided. You can check out the `test_ontheflydataset.py` test file to see how it is used in principle. 