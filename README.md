This repository hosts the code associated with the NeurIPS 2022 paper [Subgroup Robustness Grows on Trees: An Empirical Baseline Investigation](https://arxiv.org/abs/2211.12703). If you use this code in your research, please cite our paper.

The repository contains the following directories:

* `configs`: contains `.yaml` files used to define hyperparameters of models.
* `scripts`: contains scripts to download data and `train.py`, the script used to launch model training.
* `src`: the core libraries for dataset loading, preprocessing, training, and evaluation.
* `sweeps`: contains `.yaml` files used to define Weights & Biases "sweeps" to replicate the experiments.

## Environment Setup

It is recommended to use a virtual environment when running the code in this repository. Two types of virtual environments are supported, `venv` and `conda`. We recommend `venv` due to speed and ease of use, but `conda` may be a better cross-platform solution.

### Option 1: pip + venv (recommended)

To set up an environment using Python's native virtual environment tool `venv`, run the following:

``` 
python3.7 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
pip install -e .
```

You can then activate the virtual environment with `source activate venv`.

### Option 2: conda

Use the included `environment.yaml` to set up a conda environment:

```
conda env create -f environment.yaml
```

You can then activate the virtual environment with `conda activate srgot` (srgot = Subgroup Robustness Grows On Trees).

Whichever option you choose, ensure that the environment is activated while running the code samples below.

## Datasets

We are not able to host the datasets directly in this repository, but all datasets are publicly available (with the exception of LARC, see below). We provide a script to download the Adult, German Credit, COMPAS, and Communities and Crime datasets at `scripts/download.sh`. The ACS datasets (Income, Public Coverage) are downloaded at runtime by `folktables`.

The BRFSS dataset can be dowloaded from [Kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system); place the CSV data files at `datasets/brfss`.

Information on accessing the LARC dataset is [here](https://enrollment.umich.edu/data/learning-analytics-data-architecture-larc).

## Model Training

Individual model training runs are executed by executing `scripts/train.py` with the appropriate hyperparameter configurations. 

As an example, to train a model using the default random forest hyperparameters on the Communities and Crime dataset, run the following from within the virtual environment:

``` 
python scripts/train.py \
--default_config configs/defaults/random_forest.yaml \
--dataset candc
```

The `train.py` script uses the following two important flags, shown above:
* `default_config` provides a path to a .yaml file specifying all of the required named hyperparameters for the model (these are defined for each model in `src/config.py`).
* `dataset` gives the name of the dataset. Valid dataset names are: `adult`, `brfss`, `candc`, `compas`, `german`, `income`, `larc-grade`, `pubcov`.

## Hyperparameter sweeps

Hyperparameter sweeps can be conducted using the provided `yaml` files (in the `sweeps` directory) via `wandb`. Hyperparameter sweeps are grid sweeps, which means that a complete training run will be executed for every hyperparameter configuration defined in the sweep file.

To initiate a sweep:

Set the environment variable `WANDB_API_KEY` (this may be set for you by weights & biases, or you may be probmped to set it). Then, initiate the sweep by running e.g.:

```wandb sweep sweeps/adult/xgboost.yaml```

Follow the instructions that appear after the prompt to start one or more "agents" to do the sweeping in a separate shell. For scripts we used to launch several agents at once, please reach out to the repo authors.

*Note: you may encounter performance issues in the Weights & Biases interface when running the full hyperparameter sweeps in this repo. See note below.*

## Questions and Issues

If you encounter a problem with the code or have a question, we will do our best to promptly address it. Please file an [issue](https://github.com/jpgard/subgroup-robustness-grows-on-trees/issues) in Github.


## A note on Weights & Biases

The scripts use [Weights & Biases](https://wandb.ai/) by default. However, as of publication time, the Weights & Biases backend is *not* designed to handle hyperparameter sweeps as large as the one tracked in this project. As such, we don't recommend using a single w&b project to store all of your sweeps, if you decide to run the complete set of sweeps to replicate the paper (around 341k runs). Running only a subset of those sweeps within a single project is fine in our experience (under ~10k runs). 

We thus advise sending each sweep to a separate wandb project. Even then, you will likely experience poor performance when fetching results from the wandb server due to the size and number of the sweeps in the project -- sorry, we have no suggestions to improve this performance, as this is a limitation of Weights and Biases as of the time of publication.