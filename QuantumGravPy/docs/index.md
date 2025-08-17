# Welcome to the QuantumGrav documentation!

This project is dedicated to providing tools for creating causal sets as used in the corresponding approach to quantum gravity, and to building machine learning systems for analyzing them. Therefore, this project consists of two parts: 

- A Julia package called `QuantumGrav.jl` which is build on top of [`CausalSets.jl`](https://www.thphys.uni-heidelberg.de/~hollmeier/causalsets/), which creates a causal sets of a different varieties (manifold-like, random non-manifold like, non-causal-set DAGs). For now, manifold-like causal sets are restricted to 2D. This package also provides functions for deriving a set of quantities from the graph-level properties of the produced causal sets. It also allows for storing the data in `HDF5` or `Zarr` files. 

- A Python package called `QuantumGravPy` which is based on [`pytorch-geometric`](https://pytorch-geometric.readthedocs.io/en/latest/), [`h5py`](https://www.h5py.org/) and [`zarr`](https://zarr.readthedocs.io/en/stable/). This package  is thus responsible for the data preprocessing, and model training. This package is based on a configuration-code separation in which you will define your model using YAML files and only supply code where the supplied abstractions do not suffice. 

Start with the [Getting started](./getting_started.md) page to get up and running. 

