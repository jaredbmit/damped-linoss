<h1 align='center'> Learning to Dissipate Energy in Oscillatory State-Space Models <br></h1>

This repository implements Damped Linear Oscillatory State-Space Models (D-LinOSS), an expressive and efficient extension to the original LinOSS model. This codebase is adapted from the <a href="https://github.com/tk-rusch/linoss">linoss</a> and <a href="https://github.com/Benjamin-Walker/log-neural-cdes">log-neural-cdes</a> repositories.

---

## Requirements

This repository is implemented in python 3.10 and uses Jax as the machine learning framework.

### Environment

This project uses `uv` as the Python package manager and environment tool.

Installation:
```
curl -Ls https://astral.sh/uv/install.sh | sh
```

Configuring the damped-linoss environment:
```
cd damped-linoss/
uv sync
```
This will create a virtual environment in `linoss/.venv`.

Use `uv run` instead of `python` when running scripts.

If running `scripts/process_uea.py` throws this error: No module named 'packaging'
Then run: `uv pip install packaging`

---

## Data

The folder `scripts` contains the scripts for downloading data, preprocessing the data, and creating dataloaders and datasets. Raw data should be downloaded into the `data/raw` folder. Processed data should be saved into the `data/processed` folder in the following format: 
```
processed/{collection}/{dataset_name}/data.pkl, 
processed/{collection}/{dataset_name}/labels.pkl,
processed/{collection}/{dataset_name}/original_idxs.pkl (if the dataset has original data splits)
```
where data.pkl and labels.pkl are jnp.arrays with shape (n_samples, n_timesteps, n_features) and (n_samples, n_classes) respectively. If the dataset had original_idxs then those should be saved as a list of jnp.arrays with shape [(n_train,), (n_val,), (n_test,)].

### The UEA Datasets

The UEA datasets are a collection of multivariate time series classification benchmarks. They can be downloaded by running `scripts/download_uea.py` and preprocessed by running `scripts/process_uea.py`.

### The PPG-DaLiA Dataset

The PPG-DaLiA dataset is a multivariate time series regression dataset, where the aim is to predict a person’s heart rate using data collected from a wrist-worn device. The dataset can be downloaded from the <a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">UCI Machine Learning Repository</a>. The data should be unzipped and saved in the `data/raw` folder in the following format `PPG_FieldStudy/S{i}/S{i}.pkl`. The data can be preprocessed by running the `scripts/process_ppg.py` script.

---

## Experiments

The code for training and evaluating the models is contained in `linoss/train.py`. Experiments can be run using the `run_experiment.py` script. This script requires you to specify a folder containing hyperparameter spreads for a given experiment. These experiment folders can be generated using `create_experiment.py`.

To create a set of experiments, manually define the grid or random search by editing the file `src/damped_linoss/scripts/create_experiment.py`. Then, run:
```
uv run python -m damped_linoss.scripts.create_experiment
```

To run an experiment:
```
uv run python -m damped_linoss.scripts.run_experiments --experiment_folder <path/to/experiment>
```

To view the outputs of an experiment:
```
uv run python -m damped_linoss.scripts.process_results.py <path/to/experiment>
```
