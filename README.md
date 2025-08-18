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

- install the dependencies. Because these differ based on hardware, please refer to the [pytorch-geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for further details.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
``` 

See [the pytorch installation page](https://pytorch.org/get-started/locally/) and the [pytorch-geometric installation page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), respectively, for more information

- finally, install the package:

```bash 
```bash 
cd ./QuantumGravPy
python3 -m pip install .
``` 

- or, for development, do: 
```bash 
cd ./QuantumGravPy
python3 -m pip install -e .[dev]
```

### Build the documentation 
You need to do a development installation as described above: 
```bash 
cd ./py
python3 -m pip install -e .[dev]
```

This will install the mkdocs package. Then, in the `QuantumGravPy` directory, run 
```bash 
mkdocs serve
```
The output will contain a message at the end: 
```bash 
INFO    -  [13:48:53] Serving on http://127.0.0.1:8000/
```

Opening `http://127.0.0.1:8000/` in your browser will then display the documentation. Currently, we only have python documentation. 

## Julia
- To work on the software, clone the repo, and in the base directory in the terminal run: 
- `julia`
- hit `]` 
- type `develop .` 
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