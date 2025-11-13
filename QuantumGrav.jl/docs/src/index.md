# QuantumGrav.jl

## Getting Started 
This package provides functionality to generate csets of different types. Currently, it only supports 2D causal sets in general.
The package builds on [CausalSets.jl](https://www.thphys.uni-heidelberg.de/~hollmeier/causalsets/).

Currently supported are:
- simply connected manifold like causal sets based on analytical manifolds 
- manifold like csets with complex, branched connections 
- random: Causal sets with randomly selected links 
- layered: Causal sets in which events are ordered in layers along the time dimension.
- merged: A merger of a random and manifold like causal set 
- destroyed: A manifold like causal set with some edges being filpped 
- grid like: Causal sets in which events are ordered in a grid like fashion according to a certain scheme, which can be one of 'quadratic', 'rectangular', 'rhombic', 'hexagonal', 'oblique'

You currently have to install the package from the github repository directly: 
```julia
import Pkg 
Pkg.add(url = "https://github.com/ssciwr/QuantumGrav.git", subdir="QuantumGrav.jl")
```

##  Usage 
The main entry point for cset generation is the `CsetFactory` struct. This is constructed from a configuration dictionary that can be read from disk.
The configuration can be stored in any kind of file that results in a nested dictionary as described below, but YAML is the most common and typically prefered format. 

```julia 
import QuantumGrav as QG 
import YAML
config = _YAML.load_file(joinpath("path", "to", "configfile.yaml")) # load from yaml file. 
csetfactory = QG.CsetFactory(config)
```

Once an instance of this struct is created, you can construct a new cset via: 

```julia 
import Random 
rng = Random.Xoshiro()
cset = csetfactory((
    "random", # a cset type as described above
    1000, # number of events in the cset
))
```
This will return a causal set of the requested size and kind of the type `CausalSets.BitArrayCauset`. 

## Configuration
The causal set factories each come with a specific configuration dictionary that is validated by JSON schemas. Each cset type requires specific distribution parameters that control the stochastic generation process.

### Top-Level Configuration

The main `CsetFactory` configuration requires:

- **`seed`** (integer): Random seed for reproducibility
- **`num_datapoints`** (integer, ≥0): Number of data points to generate
- **`csetsize_distr`** (string): Name of distribution for causet sizes (e.g., "DiscreteUniform")
- **`csetsize_distr_args`** (array of integers): Arguments for the size distribution
- **`csetsize_distr_kwargs`** (object, optional): Keyword arguments for the size distribution
- **`cset_type`** (string or array of strings): Type(s) of causets to generate (see below)
- **`output`** (string): Output file path

Plus nested configurations for each cset type (see below).

### Polynomial Manifold Configuration (`polynomial`)

Required fields:
- **`order_distribution`** (string): Distribution name for polynomial order (must yield integers)
- **`order_distribution_args`** (array of integers): Distribution arguments
- **`order_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`r_distribution`** (string): Distribution name for exponential decay parameter
- **`r_distribution_args`** (array of numbers): Distribution arguments (must be > 1)
- **`r_distribution_kwargs`** (object, optional): Distribution keyword arguments

**Example:**
```julia
config["polynomial"] = Dict(
    "order_distribution" => "DiscreteUniform",
    "order_distribution_args" => [3, 10],
    "order_distribution_kwargs" => Dict(),
    "r_distribution" => "Uniform",
    "r_distribution_args" => [1.5, 3.0],
    "r_distribution_kwargs" => Dict()
)
```

### Layered Configuration (`layered`)

Required fields:
- **`connectivity_distribution`** (string): Distribution for connectivity probability between layers
- **`connectivity_distribution_args`** (array of numbers): Distribution arguments (should be in [0,1])
- **`connectivity_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`stddev_distribution`** (string): Distribution for Gaussian standard deviation in layer sizes
- **`stddev_distribution_args`** (array of numbers): Distribution arguments
- **`stddev_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`layer_distribution`** (string): Distribution for number of layers
- **`layer_distribution_args`** (array of integers): Distribution arguments
- **`layer_distribution_kwargs`** (object, optional): Distribution keyword arguments

**Example:**
```julia
config["layered"] = Dict(
    "connectivity_distribution" => "Uniform",
    "connectivity_distribution_args" => [0.3, 0.7],
    "connectivity_distribution_kwargs" => Dict(),
    "stddev_distribution" => "Uniform",
    "stddev_distribution_args" => [5.0, 20.0],
    "stddev_distribution_kwargs" => Dict(),
    "layer_distribution" => "DiscreteUniform",
    "layer_distribution_args" => [3, 10],
    "layer_distribution_kwargs" => Dict()
)
```

### Random (Connectivity-Based) Configuration (`random`)

Required fields:
- **`connectivity_distribution`** (string): Distribution for target connectivity ratio
- **`connectivity_distribution_args`** (array of numbers): Distribution arguments (should be in (0,1])
- **`connectivity_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`max_iter`** (integer, ≥1): Maximum MCMC iterations per attempt
- **`num_tries`** (integer, ≥1): Number of attempts before giving up
- **`abs_tol`** (number or null): Absolute tolerance for convergence (use `null` if using `rel_tol`)
- **`rel_tol`** (number or null): Relative tolerance for convergence (use `null` if using `abs_tol`)

**Note:** Exactly one of `abs_tol` or `rel_tol` should be non-null.

**Example:**
```julia
config["random"] = Dict(
    "connectivity_distribution" => "Uniform",
    "connectivity_distribution_args" => [0.1, 0.9],
    "connectivity_distribution_kwargs" => Dict(),
    "max_iter" => 10000,
    "num_tries" => 3,
    "abs_tol" => 0.01,
    "rel_tol" => nothing
)
```

### Destroyed Configuration (`destroyed`)

Required fields:
- **`order_distribution`** (string): Distribution for polynomial order (integers)
- **`order_distribution_args`** (array of integers): Distribution arguments
- **`order_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`r_distribution`** (string): Distribution for exponential decay parameter
- **`r_distribution_args`** (array of numbers): Distribution arguments
- **`r_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`flip_distribution`** (string): Distribution for fraction of edges to flip
- **`flip_distribution_args`** (array of numbers): Distribution arguments (typically in [0,1])
- **`flip_distribution_kwargs`** (object, optional): Distribution keyword arguments

**Example:**
```julia
config["destroyed"] = Dict(
    "order_distribution" => "DiscreteUniform",
    "order_distribution_args" => [3, 8],
    "order_distribution_kwargs" => Dict(),
    "r_distribution" => "Uniform",
    "r_distribution_args" => [1.5, 2.5],
    "r_distribution_kwargs" => Dict(),
    "flip_distribution" => "Uniform",
    "flip_distribution_args" => [0.05, 0.3],
    "flip_distribution_kwargs" => Dict()
)
```

### Grid Configuration (`grid`)

Required fields (similar to polynomial, plus grid-specific):
- **`grid_distribution`** (string): Distribution for grid type index
- **`grid_distribution_args`** (array of integers): Distribution arguments
- **`grid_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`rotate_distribution`** (string): Distribution for rotation angle (degrees)
- **`rotate_distribution_args`** (array of numbers): Distribution arguments
- **`rotate_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`order_distribution`**, **`r_distribution`**: Same as polynomial configuration

Grid types: 'quadratic', 'rectangular', 'rhombic', 'hexagonal', 'oblique'

### Merged Configuration (`merged`)

Required fields:
- **`order_distribution`**, **`r_distribution`**: For the manifold component
- **`link_prob_distribution`** (string): Distribution for cross-insertion link probability
- **`link_prob_distribution_args`** (array of numbers): Distribution arguments (in [0,1])
- **`link_prob_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`n2_rel_distribution`** (string): Distribution for relative size of inserted component
- **`n2_rel_distribution_args`** (array of numbers): Distribution arguments
- **`n2_rel_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`connectivity_distribution`** (string): Distribution for connectivity of inserted random component
- **`connectivity_distribution_args`** (array of numbers): Distribution arguments
- **`connectivity_distribution_kwargs`** (object, optional): Distribution keyword arguments

### Complex Topology Configuration (`complex_topology`)

Required fields:
- **`order_distribution`**, **`r_distribution`**: For the base manifold
- **`vertical_cut_distribution`** (string): Distribution for number of vertical (timelike) cuts
- **`vertical_cut_distribution_args`** (array of numbers): Distribution arguments
- **`vertical_cut_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`finite_cut_distribution`** (string): Distribution for number of finite (mixed) cuts
- **`finite_cut_distribution_args`** (array of numbers): Distribution arguments
- **`finite_cut_distribution_kwargs`** (object, optional): Distribution keyword arguments
- **`tol`** (number): Floating point tolerance for geometric comparisons

**Example:**
```julia
config["complex_topology"] = Dict(
    "order_distribution" => "DiscreteUniform",
    "order_distribution_args" => [3, 8],
    "order_distribution_kwargs" => Dict(),
    "r_distribution" => "Uniform",
    "r_distribution_args" => [1.5, 2.5],
    "r_distribution_kwargs" => Dict(),
    "vertical_cut_distribution" => "DiscreteUniform",
    "vertical_cut_distribution_args" => [0, 3],
    "vertical_cut_distribution_kwargs" => Dict(),
    "finite_cut_distribution" => "DiscreteUniform",
    "finite_cut_distribution_args" => [0, 2],
    "finite_cut_distribution_kwargs" => Dict(),
    "tol" => 1e-12
)
```

### Distribution Specification

All distributions use the pattern:
- `<name>_distribution`: String name of distribution from `Distributions.jl` (e.g., "Uniform", "Normal", "DiscreteUniform")
- `<name>_distribution_args`: Positional arguments as array (e.g., `[min, max]` for Uniform)
- `<name>_distribution_kwargs`: Optional keyword arguments as dictionary

Common distributions:
- **Uniform(a, b)**: Continuous uniform on [a, b]
- **Normal(μ, σ)**: Normal with mean μ and std σ
- **DiscreteUniform(a, b)**: Discrete uniform on {a, a+1, ..., b}
- **Beta(α, β)**: Beta distribution (useful for probabilities)

## API 

A Julia package for generating and manipulating causal sets for quantum gravity research.

### Causal Set Generation 

##### Manifold-like Causal Sets 

Generate causal sets by sprinkling points into polynomial manifolds:

```@docs
make_polynomial_manifold_cset
```

##### Layered Causal Sets 

Generate layered causal sets with controlled connectivity between layers:

```@docs
gaussian_dist_cuts
create_random_layered_causet
```

##### Connectivity-Based Sampling

Sample causal sets targeting specific connectivity values:

```@docs
sample_bitarray_causet_by_connectivity
random_causet_by_connectivity_distribution
```

##### Grid-like Causal Sets 

Generate regular grid structures in causal sets:

```@docs
generate_grid_from_brillouin_cell
generate_grid_2d
create_grid_causet_2D
create_grid_causet_2D_polynomial_manifold
```

##### Branched Manifold Causal Sets 

Generate causal sets with complex topological features:

```@docs
BranchedManifoldCauset
make_branched_manifold_cset
```

##### Merged Causal Sets

Merge and insert causal sets:

```@docs
merge_csets
insert_cset
insert_KR_into_manifoldlike
```

##### Destroyed Causal Sets 

Destroy manifold-like structure through random edge flips:

```@docs
destroy_manifold_cset
```

### Curvature Analysis

Compute curvature on manifold-like causal sets:

```@docs
Ricci_scalar_2D
Ricci_scalar_2D_of_sprinkling
```

### Causal Set Factories

Factory pattern for generating different types of causal sets:

```@docs
PolynomialCsetMaker
RandomCsetMaker
LayeredCsetMaker
DestroyedCsetMaker
GridCsetMakerPolynomial
MergedCsetMaker
ComplexTopCsetMaker
CsetFactory
encode_csettype
```

### Data Storage and Preparation

Tools for data production and storage:

```@docs
prepare_dataproduction
copy_sourcecode
get_git_info!
dict_to_zarr
```

### Graph Utilities 

Low-level graph operations:

```@docs
make_adj
max_pathlen
transitive_reduction!
```

### Helper Functions

```@docs
make_pseudosprinkling
validate_config
``` 