# QuantumGrav
Quantum gravity project experimental repo. 

## Python
*this assumes a UNIX system* 

- Set up a virtual environment first
```bash 
python3 -m venv .venv 
```
- activate it
```bash
source ./.venv/bin/activate  
```

- install the dependencies. Because these differ based on hardware, there are 
`requirements.txt` files for installation. 
```bash 
python3 -m pip install -r requirements_torch_cu128.txt
```
for torch itself and 
```bash 
python3 -m pip install -r requirements_torchgeo_cu128.txt
```
for pytorch-geometric for example. These will install the versions for cuda 12.8. 
This can also be done by hand: 

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
``` 

See [the pytorch isntallation page](https://pytorch.org/get-started/locally/) and the [pytorch-geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), respectively, for mor information 

- to install a specific version, modify the `requirements_torch_*` files to contain the right versions and urls for your hardware and OS.

```bash 
- finally, install the package 
```bash 
python3 -m pip install .
``` 

- or, for development, do: 
```bash 
python3 -m pip install -e .[dev]
```






## Julia
- To work on the software, clone the repo, and in the base directory in the terminal run: 
- `julia`
- hit `]` 
- type `activate .` 
- hit ctrl+c to get out of the package manager
- `exit()` to leave julia

- To add the dependency for data generation: 
```julia 
using Pkg 
Pkg.add("CausalSets")
```

- To run tests:
  - navigate to the base directory of the package 
  - open julia, then run: 
    - hit `]` 
    - type `activate .` 
    - hit ctrl+c or backspace to leave the package manager 
    - run: 
    ```julia 
    using TestItemRunner
    @run_package_tests
    ``` 
    - `exit()` to leave julia
- this will give you more detailed output. 
- Alternatively, you can also type `test` in the package manager environment 
to run the tests
you can also use vscode to run the t