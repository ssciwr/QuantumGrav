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
- destroyed: A manifold like causal set with some edges being flipped, which makes it a non-manifold like causal set.
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
cset = csetfactory((
    "random", # a cset type as described above
    1000, # number of events in the cset
))
```
This will return a causal set of the requested size and kind of the type `CausalSets.BitArrayCauset`.

This package uses the zarr format for storing data to file. From a constructed cset, a multitude of observables can be derived.
These can be stored in a dictionary that then can be written directly into a zarr file. Secondly, a helper function is provided to copy the config and the source code used to generate the observables to the same directory as the data and creates a Zarr `DirectoryStore` there. This helps with reproducibility and documentation of different data generation runs. The created Zarr file will contain the pid of the generating process and the date and time in `yyyy-mm-dd_HH-MM-SS` format.


Any code that generates csets first must define a function that takes a Causal Set factory and builds a dictionary of computed observables:

```julia
import QuantumGrav as QG
import Random
import Zarr

function make_cset_data(csetfactory)
    cset = csetfactory(
        "random", # a cset type as described above
        1000, # number of events in the cset
    )

    return Dict(
        "observable1" => make_observable1(cset) # assume the `make_observable1` would be user defined
        "observable2" => make_observable2(cset)
    )
end
```

Then we load the config and call the `prepare_dataproduction` function to set everything up. The second argument will serve as an anchor to determine which source code files shall be copied over. In this case, it will be the file where the `make_cset_data` function is defined. If you add other functions, their respective sourcecode files will be copied as well.

```julia
config = _YAML.load_file(joinpath("path", "to", "configfile.yaml")) # load from yaml file.

path_to_store, zarr_store = QG.prepare_dataproduction(
    config,
    [make_cset_data];
    nameaddition = "random_data", # will create a folder starting with 'random_data' and ending with '.zarr'
)

# make the factory
factory = QG.CsetFactory(config)
```
QG provides a `setup_config` function for this which makes this a little easier.

Finally, we create our data. Unfortunately, because this writes metadata, it can't be
parallelized within the same store or group.

```julia
for i in 1:num_csets
    data = Dict("cset_$i"=> make_cset_data(factory))
    QG.dict_to_zarr(file, data)
end
```
The whole procedure will result in a directory of the form `output/*csettype*_{pid}_{yyyy-mm-dd_HH-MM-SS}.zarr`, which contains
directories `cset_{i}` and therein the respective observables.

The resulting directory structure will look like this:

```
output/
└── random_data_12345_2025-11-13_14-30-45.zarr/
    ├── .zattrs                    # Zarr metadata
    ├── .zgroup                    # Zarr group marker
    ├── config.yaml                # Copy of configuration used
    ├── make_cset_data.jl          # Source code snapshot
    ├── ...                        # other sourcecode files
    ├── cset_1/
    │   ├── .zgroup
    │   ├── observable1/           # Arrays stored as Zarr arrays
    │   │   ├── .zarray
    │   │   └── ...
    │   └── observable2/
    │       ├── .zarray
    │       └── ...
    ├── cset_2/
    │   ├── .zgroup
    │   ├── observable1/
    │   └── observable2/
    │
    ├── ...                        # more csets
    └── cset_N/
        ├── .zgroup
        ├── observable1/
        └── observable2/
```
This process has been encapsulated into the function `QuantumGrav.produce_data`.

## Examples
For examples for how to write your own variant of this workflow e.g., for multiprocessing, see the `examples` directory int QuantumGrav/QuantumGrav.jl in the repository.

### `produce_data.jl` example

This example demonstrates how to generate a small dataset using the bundled script in [examples/produce_data.jl](https://github.com/ssciwr/QuantumGrav/blob/main/QuantumGrav.jl/examples/produce_data.jl). It orchestrates multi-process and multi-threaded data generation, writes observables to a Zarr store, and cleans up worker processes afterwards.

**What it does**
- Spawns `num_workers` Julia processes, each with `num_threads` threads, optionally setting BLAS threads.
- Loads a configuration (default plus overrides) and builds a `QG.CsetFactory` on workers.
- Repeatedly generates causal sets and basic observables (adjacency, link matrix) via the function `make_cset_data`.
- Writes results to a Zarr `DirectoryStore` created by `QG.produce_data`.
- Removes worker processes on exit, even if an error occurs. This step is important when using multiprocessing to not leave orphaned processes on the system if the script crashes.

**Code sections**
Open the script on the side to follow along the different sections. From top to bottom:
- Command-line args: Parses `--config`, `--num_workers`, `--num_threads`, `--num_blas_threads`, `--chunksize`, and `--help`.
- Multiprocessing setup: Adds processes with `Distributed.addprocs`, passes `--threads` and `--project` flags.
- Environment on workers: `@everywhere` imports `QuantumGrav` and `LinearAlgebra`, and sets BLAS threads.
- Data generation helpers: Defines `cset_type_encoder` and `make_cset_data(worker_factory)` to produce minimal features.
- Main run block: Calls `QG.produce_data(chunksize, configpath, make_cset_data)` inside a try/catch/finally that ensures `rmprocs`.

CLI options
- `--config <path>`: Path to a YAML or JSON config file. If omitted, defaults are used.
- `--num_workers <int>`: Number of processes to spawn (≥1).
- `--num_threads <int>`: Threads per process (≥1).
- `--num_blas_threads <int>`: Threads used by BLAS (independent of Julia threads).
- `--chunksize <int>`: Number of csets to generate per write chunk.
- `--help | -h`: Prints usage information and exits.

**Example configuration overrides**
You can define a minimal override file such as [examples/example_config.yaml](https://github.com/ssciwr/QuantumGrav/blob/main/QuantumGrav.jl/examples/example_config.yaml) to adjust selected parameters while inheriting the package defaults:

```yaml
seed: 21
output: "./example_data"
csetsize_distr: "Normal"
csetsize_distr_args: [32, 96]

# the rest will be taken over from the default config
```

**How to run**
- From the package root, launch the script with desired resources:

```bash
julia --project=QuantumGrav.jl -O3 QuantumGrav.jl/examples/produce_data.jl \
    --config QuantumGrav.jl/examples/example_config.yaml \
    --num_workers 2 \
    --num_threads 4 \
    --num_blas_threads 4 \
    --chunksize 50
```

**Tips**
- You can also run from within the `QuantumGrav.jl` project directory: `julia --project -O3 examples/produce_data.jl ...`.
- Use `--num_workers auto`-style values by scripting around detection of cores; the example expects explicit integers.
- If you only want threading, set `--num_workers 1` and adjust `--num_threads`.

**Expected outcome**
- A Zarr directory under the `output` path (from config), named like `random_data_<pid>_<yyyy-mm-dd_HH-MM-SS>.zarr`. Unless you change it, this should be in `example_data` in the directory you called the script from.
- Within it, zarr groups (e.g., `cset_1`, `cset_2`, …) each containing basic observables produced by `make_cset_data`.
- The store includes metadata (`.zattrs`) and copied source/config via `QG.prepare_dataproduction`, aiding reproducibility.

If any generation attempt for csets fails transiently, the script retries up to 20 times per cset before raising an error.


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

Grid types: 'quadratic', 'rectangular', 'rhombic', 'hexagonal', 'oblique'. Each grid type has their own parameters.

**Example**
```julia
config["grid"] = Dict(
        "grid_distribution" => "DiscreteUniform",
        "grid_distribution_args" => [1, 6],
        "grid_distribution_kwargs" => Dict(),
        "rotate_distribution" => "Uniform",
        "rotate_distribution_args" => [0.0, 180.0],
        "rotate_distribution_kwargs" => Dict(),
        "order_distribution" => "DiscreteUniform",
        "order_distribution_args" => [2, 8],
        "order_distribution_kwargs" => Dict(),
        "r_distribution" => "Uniform",
        "r_distribution_args" => [4.0, 8.0],
        "r_distribution_kwargs" => Dict(),
        "quadratic" => Dict(), # nothing needed here
        "rectangular" => Dict(
            "segment_ratio_distribution" => "Beta",
            "segment_ratio_distribution_args" => [2.0, 1.2],
            "segment_ratio_distribution_kwargs" => Dict(),
        ),
        "rhombic" => Dict(
            "segment_ratio_distribution" => "Uniform",
            "segment_ratio_distribution_args" => [0.5, 5.5],
            "segment_ratio_distribution_kwargs" => Dict(),
        ),
        "hexagonal" => Dict(
            "segment_ratio_distribution" => "Normal",
            "segment_ratio_distribution_args" => [2.0, 0.5],
            "segment_ratio_distribution_kwargs" => Dict(),
        ),
        "triangular" => Dict(
            "segment_ratio_distribution" => "Normal",
            "segment_ratio_distribution_args" => [3.3, 1.2],
            "segment_ratio_distribution_kwargs" => Dict(),
        ),
        "oblique" => Dict(
            "segment_ratio_distribution" => "Logistic",
            "segment_ratio_distribution_args" => [2.0, 1.0],
            "segment_ratio_distribution_kwargs" => Dict(),
            "oblique_angle_distribution" => "Normal",
            "oblique_angle_distribution_args" => [45.0, 15.0],
            "oblique_angle_distribution_kwargs" => Dict(),
        ),
    )
```

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

**Example**
```julia
config["merged"] = Dict(
            "order_distribution" => "DiscreteUniform",
            "order_distribution_args" => [2, 8],
            "order_distribution_kwargs" => Dict(),
            "r_distribution" => "Normal",
            "r_distribution_args" => [4.0, 2.0],
            "r_distribution_kwargs" => Dict(),
            "n2_rel_distribution" => "Uniform",
            "n2_rel_distribution_args" => [0., 1.0],
            "n2_rel_distribution_kwargs" => Dict(),
            "connectivity_distribution" => "Beta",
            "connectivity_distribution_args" => [0.5, 0.1],
            "connectivity_distribution_kwargs" => Dict(),
            "link_prob_distribution" => "Normal",
            "link_prob_distribution_args" => [2.0, 1.5],
            "link_prob_distribution_kwargs" => Dict(),
)
```

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
- `<name>_distribution`: String name of a univariate distribution from `Distributions.jl` (e.g., "Uniform", "Normal", "DiscreteUniform")
- `<name>_distribution_args`: Positional arguments as array (e.g., `[min, max]` for Uniform)
- `<name>_distribution_kwargs`: Optional keyword arguments as dictionary. Usually not needed.

Common distributions:
- **Uniform(a, b)**: Continuous uniform on [a, b]
- **Normal(μ, σ)**: Normal with mean μ and std σ
- **DiscreteUniform(a, b)**: Discrete uniform on {a, a+1, ..., b}

See the [documentation of Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/) for all available distributions and their needed parameters

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