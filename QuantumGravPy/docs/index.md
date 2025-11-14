# Welcome to the QuantumGrav documentation!

This project is dedicated to providing tools for creating causal sets as used in the corresponding approach to quantum gravity, and to building machine learning systems for analyzing them. Therefore, this project consists of two parts:

- A Julia package called `QuantumGrav.jl` which is build on top of [`CausalSets.jl`](https://www.thphys.uni-heidelberg.de/~hollmeier/causalsets/), which creates a causal sets of a different varieties (manifold-like, random non-manifold like, non-causal-set DAGs). For now, manifold-like causal sets are restricted to 2D. This package also provides functions for deriving a set of quantities from the graph-level properties of the produced causal sets. It also allows for storing the data in `HDF5` or `Zarr` files. If you want to learn more about it, check out the [documentationh of the Julia package]((https://img.shields.io/badge/docs-julia-purple?logo=julia)](https://ssciwr.github.io/QuantumGrav/julia/)

- A Python package called `QuantumGravPy` which is based on [`pytorch-geometric`](https://pytorch-geometric.readthedocs.io/en/latest/) and [`zarr`](https://zarr.readthedocs.io/en/stable/). This package  is thus responsible for the data preprocessing, and model training. This package is based on a configuration-code separation in which you will define your model using YAML files and only supply code where the predefined abstractions do not suffice.

Start with the [Getting started](./getting_started.md) page to get up and running.

For the Python package `QuantumGravPy`, the [`Datasets and Preprocessing`](./datasets_and_preprocessing.md) section will show you how to use the supplied dataset classes for processing your raw data. Next, you should learn about the model architecture used in this package in [`Graph Neural Network models`](./models.md).
To learn how to train a model, check out the [`Model training`](./training_a_model.md) section.
Finally, the [`API documentation`](./api.md) will tell you everything you need to know about the source code of the package.


Note that the two packages are designed to function in unison, with the Julia package producing data that the python package consumes.