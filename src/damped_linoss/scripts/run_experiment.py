"""
This script serves as a main entrypoint to training models, executing (optionally in batched fashion)
a series of training runs as specified by an experiment folder.

One training run within an experiment corresponds to one folder in your file system, containing a 
`hyperparameters.yaml` file and storing all outputs, logs, and model checkpoints for that particular run.

Begin by generating a set of empty run folders containing the desired hyperparameter spread using
`create_experiment.py`. Then run this script with the --experiment_folder flag, which will iterate 
through subfolders and execute corresponding training runs.

Experiment outputs can be postprocessed with `postprocess_results.py`.

Batching splits the experiment workload over $num_tasks$ different processes, assuming this script is 
launched $num_tasks$ different times, independently, with varying values of $task_id$.
"""
import os
import yaml
import equinox as eqx
import argparse

from damped_linoss.train import create_dataset_model_and_train


def run_experiments(
    experiment_folder: str,
    task_id: int = 0,
):
    """
    Runs a series of training experiments.
    Iterates over all run subfolders in run_folder/

    Args:
        experiment_folder (str): Absolute path to parent directory containing all run subfolders.
        task_id (int): Process ID -- for batching. Defaults to 0.
        num_tasks (int): Number of processes -- for batching. Defaults to 1.
    """
    run_folder = os.path.join(experiment_folder, f"run_{int(task_id):03d}")
    hyperparameters_path = os.path.join(run_folder, "hyperparameters.yaml")
    with open(hyperparameters_path, "r") as f:
        hyperparameters = yaml.safe_load(f)

    # Train model
    model, state = create_dataset_model_and_train(run_folder, hyperparameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Absolute path to parent directory containing all subfolders to be run."
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="batching: id number of process",
    )
    args = parser.parse_args()

    run_experiments(
        args.experiment_folder,
        args.task_id,
    )
