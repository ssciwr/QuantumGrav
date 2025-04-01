# QuantumGrav
Quantum gravity project experimental repo. 

- To work on the software, clone the repo, and in the base directory in the terminal run: 
- `julia`
- hit `]` 
- type `activate .` 
- hit ctrl+c to get out of the package manager
- `exit()` to leave julia

- To add the dependency for data generation: 
```julia 
using Pkg 
Pkg.add(url = "https://codeberg.org/cyclopentane/CausalSets.jl")
```

- To run tests:
  - navigate to the base directory of the package 
  - open julia, then run: 
    - hit `]` 
    - type `activate .` 
    - hit ctrl+c to leave the package manager 
    - run: 
    ```julia 
    using TestItemRunner
    @run_package_tests
    ``` 
    - `exit()` to leave julia
