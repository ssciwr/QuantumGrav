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

```bash 
- finally, install the package 
```bash 
cd ./py
python3 -m pip install .
``` 

- or, for development, do: 
```bash 
cd ./py
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

## Hyperparamter optimization

We currently use [Sweep](https://docs.wandb.ai/guides/sweeps/) from [Weights & Biases](https://wandb.ai/site) to handle hyperparameter tuning.

First, you need to create a W&B account, then copy your API key from [wandb.ai/authorize](https://wandb.ai/authorize)

The set the `WANDB_API_KEY` environment variable
```
export WANDB_API_KEY=<your_api_key>
```

To authenticate your machine with W&B, you can use command line or Python code as instructed in [W&B Docs](https://docs.wandb.ai/quickstart/#install-the-wandb-library-and-log-in)