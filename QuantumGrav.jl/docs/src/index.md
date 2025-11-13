# QuantumGrav.jl 

A Julia package for generating and manipulating causal sets for quantum gravity research.

## Causal Set Generation 

### Manifold-like Causal Sets 

Generate causal sets by sprinkling points into polynomial manifolds:

```@docs
make_polynomial_manifold_cset
```

### Layered Causal Sets 

Generate layered causal sets with controlled connectivity between layers:

```@docs
gaussian_dist_cuts
create_random_layered_causet
```

### Connectivity-Based Sampling

Sample causal sets targeting specific connectivity values:

```@docs
sample_bitarray_causet_by_connectivity
random_causet_by_connectivity_distribution
```

### Grid-like Causal Sets 

Generate regular grid structures in causal sets:

```@docs
generate_grid_from_brillouin_cell
generate_grid_2d
create_grid_causet_2D
create_grid_causet_2D_polynomial_manifold
```

### Branched Manifold Causal Sets 

Generate causal sets with complex topological features:

```@docs
BranchedManifoldCauset
make_branched_manifold_cset
```

### Merged Causal Sets

Merge and insert causal sets:

```@docs
merge_csets
insert_cset
insert_KR_into_manifoldlike
```

### Destroyed Causal Sets 

Destroy manifold-like structure through random edge flips:

```@docs
destroy_manifold_cset
```

## Curvature Analysis

Compute curvature on manifold-like causal sets:

```@docs
Ricci_scalar_2D
Ricci_scalar_2D_of_sprinkling
```

## Causal Set Factories

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

## Data Storage and Preparation

Tools for data production and storage:

```@docs
prepare_dataproduction
copy_sourcecode
get_git_info!
dict_to_zarr
```

## Graph Utilities 

Low-level graph operations:

```@docs
make_adj
max_pathlen
transitive_reduction!
```

## Helper Functions

```@docs
make_pseudosprinkling
validate_config
``` 