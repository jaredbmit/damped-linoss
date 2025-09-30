"""
Use this script to generate many run subfolders in an experiment directory.

It's not automated very intelligently, you will need to specify which parameters should be
iterated over and which ones are static, as well as filling in all other default hyperparameter values.

An example is provided for generating D-LinOSS hyperparameter sweeps, but note that additional hyperparameters
may be necessary for other models, and some of these hyperparameters are only relevant for LinOSS. To see a list
of the model-specific hyperparameters, look in `linoss/models/generate_model.py`.
"""
import os
import yaml
import itertools
import numpy as np


def create_grid_experiment(experiment_folder, model_name, dataset_name):
    # Hyperparameter sweep
    seed = [0, 1, 2, 3, 4]
    lr = [1e-4]
    state_dim = [128]
    hidden_dim = [128]
    num_blocks = [2, 4, 6]
    include_time = [False]

    combos = itertools.product(seed, lr, state_dim, hidden_dim, num_blocks, include_time)

    for i, (_seed, _learning_rate, _state_dim, _hidden_dim, _num_blocks, _include_time) in enumerate(combos):
        hyperparameters = {
            "seed": _seed,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "data_dir": "data",
            "classification": False,
            "use_presplit": True,
            "output_step": 1,
            "num_steps": 50000,
            "print_steps": 100,
            "batch_size": 32,
            "lr": _learning_rate,
            "ssm_lr_factor": 1.0,  # _ssm_lr_factor,
            "weight_decay": 0.0,  # _weight_decay,
            "cosine_annealing": False,  # _cosine_annealing,
            "include_time": _include_time,
            "time_duration": 1.0,
            "tanh_output": False,
            "layer_name": "DampedIMEX1",
            "num_blocks": _num_blocks,
            "state_dim": _state_dim,
            "hidden_dim": _hidden_dim,
            "initialization": "ring",
            "r_min": 0.9,  # _r_min,
            "r_max": 1.0,
            "theta_min": 0.0,
            "theta_max": np.pi,  # _theta_max,
            "A_min": 0.0,
            "A_max": 1.0,
            "G_min": 0.0,
            "G_max": 1.0,
            "dt_std": 0.5,
            "drop_rate": 0.1,
        }

        # Write config
        run_folder = experiment_folder + f"run_{i:03}/"
        os.makedirs(run_folder, exist_ok=True)
        with open(run_folder + "hyperparameters.yaml", "w") as file:
            hyperparameters = yaml.dump(hyperparameters, file)


def create_random_experiment(experiment_folder, model_name, dataset_name):
    # Hyperparameter sweep
    num_runs = 100
    learning_rate = [5e-3, 5e-5]
    state_dim = [16, 256]
    hidden_dim = [16, 256]
    num_blocks = [2, 6]
    include_time = [False, True]
    batch_size = [4, 64]
    r_min = [0.5, 1.0]
    theta_max = [np.pi/12, np.pi]
    A_max = [1.0, 32.0]
    G_max = [1.0, 32.0]
    # dt_std = [0.0, 1.0]
    # drop_rate = [0.0, 0.1]
    weight_decay = [0.0, 0.05]
    cosine_annealing = [False, True]

    for i in range(num_runs):
        _seed = int(np.random.randint(0, num_runs))
        _learning_rate = float(np.exp(np.random.uniform(np.log(learning_rate[0]), np.log(learning_rate[1]))))
        _include_time = bool(np.random.choice(include_time))
        _num_blocks = int(np.random.uniform(*num_blocks))
        # _num_blocks = np.random.randint(num_blocks[0], num_blocks[1] + 1)
        _state_dim = int(np.exp(np.random.uniform(np.log(state_dim[0]), np.log(state_dim[1]))))
        _hidden_dim = int(np.exp(np.random.uniform(np.log(hidden_dim[0]), np.log(hidden_dim[1]))))
        _batch_size = int(np.random.uniform(*batch_size))
        _r_min = float(np.random.uniform(*r_min))
        _theta_max = float(np.random.uniform(*theta_max))
        _A_max = float(np.random.uniform(*A_max))
        _G_max = float(np.random.uniform(*G_max))
        # _ds = float(np.random.uniform(*dt_std))
        # _dr = float(np.random.uniform(*drop_rate))
        # _ssm_lr_factor = float(np.random.choice(ssm_lr_factor))
        _weight_decay = float(np.random.uniform(*weight_decay))
        _cosine_annealing = bool(np.random.choice(cosine_annealing))

        hyperparameters = {
            "seed": _seed,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "data_dir": "data",
            "classification": True,
            "use_presplit": True,
            "output_step": 1,
            "num_steps": 100000,
            "print_steps": 1000,
            "batch_size": _batch_size,
            "lr": _learning_rate,
            "ssm_lr_factor": 1.0,  # _ssm_lr_factor,
            "weight_decay": 0.,  # _weight_decay,
            "cosine_annealing": False,  # _cosine_annealing,
            "include_time": _include_time,
            "time_duration": 1.0,
            "tanh_output": False,
            "layer_name": "DampedIMEX1",
            "num_blocks": _num_blocks,
            "state_dim": _state_dim,
            "hidden_dim": _hidden_dim,
            "initialization": "ring",
            "r_min": 0.9,  # _r_min,
            "r_max": 1.0,
            "theta_min": 0.0,
            "theta_max": np.pi,  # _theta_max,
            "A_min": 0.0,
            "A_max": _A_max,
            "G_min": 0.0,
            "G_max": _G_max,
            "dt_std": 0.5,
            "drop_rate": 0.1,
        }

        # Write config
        run_folder = experiment_folder + f"run_{i:03}/"
        os.makedirs(run_folder, exist_ok=True)
        with open(run_folder + "hyperparameters.yaml", "w") as file:
            hyperparameters = yaml.dump(hyperparameters, file)


if __name__ == "__main__":
    model_name = "LinOSS"
    dataset_name = "Adding500"
    experiment_folder = f"experiments/D-LinOSS-IMEX1-RT-Last/{dataset_name}/"

    if os.path.exists(experiment_folder):
        raise RuntimeError("Experiment already exists!")

    create_grid_experiment(experiment_folder, model_name, dataset_name)
    # create_random_experiment(experiment_folder, model_name, dataset_name)
                                                                           
