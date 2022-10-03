This repository hosts the code associated with the NeurIPS 2022 paper "Subgroup Robustness Grows on Trees: An Empirical Baseline Investigation."

## Environment Setup

It is recommended to use a virtual environment when using this repository. Two types of virtual environments are supported, `conda` and `venv`.

### Option 1: pip + venv (recommended)

To set up an environment using Python's native virtual environment tool `venv`, run the following:

``` 
python3.7 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
pip install -e .
```

### Option 2: conda

Use the included `environment.yaml` to set up a conda environment:

```
conda env create -f environment.yaml
```

Whichever option you choose, ensure that the environment is activated while running the code samples below.

## Datasets

We are not able to host the datasets directly in this repository, but all datasets are publicly available (with the exception of LARC, see below). We provide a script to download the Adult, German Credit, COMPAS, and Communities and Crime datasets at `scripts/download.sh`. The ACS datasets (Income, Public Coverage) are downloaded at runtime by `folktables`.

The BRFSS dataset can be dowloaded from [Kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system); place the CSV data files at `datasets/brfss`.

Information on accessing the LARC dataset is [here](https://enrollment.umich.edu/data/learning-analytics-data-architecture-larc).

## Model Training

Most models are trained by executing `scripts/train.py` with the appropriate hyperparameter configurations. 

As an example, to train a model using the default random forest hyperparameters on the Communities and Crime dataset, run the following from within the virtual environment:

``` 
python scripts/train.py \
--default_config configs/defaults/random_forest.yaml \
--dataset candc
```

For other dataset-specific experiments, such as domain transfer experiments, see the various subdirectories of `scripts/experiments`.

## Hyperparameter sweeps

Hyperparameter sweeps can be conducted using the provided yaml files and `wandb`. 

To initiate a sweep:

Set the environment variable `WANDB_API_KEY` (this may be set for you by weights & biases, or you may be probmped to set it). Then, initiate the sweep by running e.g.:

```wandb sweep sweeps/adult/xgboost.yaml```

Follow the instructions that appear after the probmp to start one or more "agents" to do the sweeping in a separate shell.

*Note: running the full hyperparameter sweeps in this repo currently stretches the limits of performance for wandb. It will be helpful to start a clean project to hold the sweeps for your experiments, but even then, you will likely experience poor performance when fetching results from the wandb server due to the size and number of the sweeps in the project.*


# A note on Weights & Biases

The scripts use [Weights & Biases](https://wandb.ai/) by default. However, as of publication time, the Weights & Biases backend is *not* designed to handle hyperparameter sweeps as large as the one tracked in this project. As such, we don't recommend using a single w&b project to store all of your sweeps, if you decide to run the complete set of sweeps to replicate the paper (around 341k runs). Running only a subset of those sweeps within a single project is fine in our experience (under ~10k runs).