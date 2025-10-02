# API Reference

## Main model class
The main model class `GNNModel` is there to tie together the Graph neural network backbone and a multilayer perceptron classifier model that can be configured for various tasks. 
::: QuantumGrav.gnn_model 
    handler: python
    options:
      show_source: true

## Graph Neural network submodels
The submodel classes in this section comprise the graph neural network backbone of a QuantumGrav model. 

### Graph model block 
This submodel is the main part of the graph neural network backbone, composed of a set of GNN layers from `pytorch-geometric` with dropout and `BatchNorm`. 
::: QuantumGrav.gnn_block  
    handler: python
    options:
      show_source: true

### Base class for models composed of linear layers
::: QuantumGrav.linear_sequential 
    handler: python
    options:
      show_source: true

## Model evaluation
This module provides base classes that take the output of applying the model to a validation or training dataset, and derive useful quantities to evaluate the model quality. These do not do anything useful by default. Rather, you must derive your own class from them that implemements your desired evaluation, e.g., using an F1 score. 

::: QuantumGrav.evaluate 
    handler: python
    options:
      show_source: true

## Datasets 
The package supports three kinds of datasets with a common baseclass `QGDatasetBase`. For the basics of how those work, check out [the pytorch-geometric documentation of dataset](https://pytorch-geometric.readthedocs.io/en/2.5.3/notes/create_dataset.html)

These are: 
- `QGDataset`: A dataset that relies on an on-disk storage of the processed data. It lazily loads csets from disk when needed. 
- `QGDatasetInMemory`: A dataset that holds the entire processed dataset in memory at once. 
- `QGDatasetOnthefly`: This dataset does not hold anything on disk or in memory, but creates the data on demand from some supplied Julia code. 

### Dataset base class
::: QuantumGrav.dataset_base 
    handler: python
    options:
      show_source: true

### Dataset holding everything in memory
::: QuantumGrav.dataset_inmemory 
    handler: python
    options:
      show_source: true

### Dataset creating csets on the fly 
::: QuantumGrav.dataset_onthefly 
    handler: python
    options:
      show_source: true

### Dataset loading data from disk
::: QuantumGrav.dataset_ondisk 
    handler: python
    options:
      show_source: true

## Julia-Python integration
This class provides a bridge to some user-supplied Julia code and converts its output into something Python can work with. 

::: QuantumGrav.julia_worker 
    handler: python
    options:
      show_source: true

## Model training
This consists of two classes, one which provides the basic training functionality - `Trainer`, and a class derived from this, `TrainerDDP`, which provides functionality for distributed data parallel training. 

### Trainer 
This class provides wrapper functions for setting up a model and for training and evaluating it. The basic concept is that everything is defined in a yaml file and handed to this class together with evaluator classes. After construction, the `train` and `test` functions will take care of the training and testing of the model. 

::: QuantumGrav.train 
    handler: python
    options:
      show_source: true

### Distributed data parallel Trainer class
This is based on [this part of the pytorch documentation](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) and is **untested** at the time of writing. 

::: QuantumGrav.train_ddp 
    handler: python
    options:
      show_source: true

## Utilities
General utilities that are used throughout this package. 

::: QuantumGrav.utils 
    handler: python
    options:
      show_source: true