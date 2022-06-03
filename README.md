
# PyBADS
`pybads` is the port of the BADS (Bayesian Adaptive Direct Search) algorithm to Python 3.x.
The reference code is the [MATLAB toolbox](https://github.com/lacerbi/bads).

## How to install and run the package (temporary)

We are using the dependencies listed in `requirements.txt`. Please list all used dependencies there.
For convenience, we also have a temporary installer in `setup.py`. Also list the used dependencies there.

The necessary packages can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or pip.

The most stable way to install and run `pybads` at the moment is:

1. Clone/update the `pybads` GitHub repo locally.
2. Create a new environment in conda: `conda create --name pybads-dev python=3.9`
3. Activate the environment: `conda activate pybads-dev`
4. From the `pybads` folder, run: `pip install -e .`
5. Install Jupyter notebook: `conda install jupyter`

If the list of requirements subsequently changes, you will only need to rerun `pip install -e .`.

### `gpyreg` package

To run `pybads` you will also need the `gpyreg` package, a lightweight Gaussian process regression library.
For now, since the package is not in a `conda` or `pip` package repository, you need to run the additional steps:

- Clone `gpyreg` from its [GitHub repo](https://github.com/lacerbi/gpyreg).
- Install `gpyreg` in the `pybads-dev` environment running `pip install -e .` from the `gpyreg` folder.

If the list of requirements subsequently changes, you will only need to rerun `pip install -e .`
### Alternative installation commands

These are alternative ways to install the required dependencies:

```
conda env create --file environment.yml
```

or

```
pip install -i requirements.txt
```

The `environment.yml` seems not to work properly in some setups (e.g., Windows), which is something to be investigated.

### Run examples ###

