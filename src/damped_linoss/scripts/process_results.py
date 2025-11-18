import os
import yaml
import glob
import numpy as np
import math
import statistics
from collections import defaultdict
import argparse


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def make_group_key(hparams, keys):
    """Create a group key string from selected hyperparameter keys."""
    parts = []
    for k in keys:
        # Support nested keys like "training.seed"
        value = hparams
        for subkey in k.split('.'):
            value = value.get(subkey, None)
            if value is None:
                break
        parts.append(f"{k}={value}")
    return ", ".join(parts)
    

def process_results(
    experiment_folder: str,
    sort_increasing: bool,
    sort_by: str="mean",
):
    groups = defaultdict(list)
    group_keys_to_use = ["model_name", "dataset_name", "lr", "state_dim", "hidden_dim", "num_blocks", "include_time", "batch_size", "r_min", "r_max", "theta_min", "theta_max", "A_min", "A_max", "G_min", "G_max", "dt_std", "drop_rate"]

    # Find all results.txt in run_XXX folders under exp_root
    pattern = os.path.join(experiment_folder, "run_*/test_metric.txt")
    for result_path in glob.glob(pattern, recursive=True):
        dir_path = os.path.dirname(result_path)
        hyper_path = os.path.join(dir_path, "hyperparameters.yaml")
        meta_path = os.path.join(dir_path, "metadata.txt")
        metric_path = os.path.join(dir_path, "log_metrics.npy")

        # Load result.txt
        try:
            with open(result_path, "r") as f:
                lines = f.readlines()
                test_metric = float(lines[0])
        except Exception as e:
            print(f"Failed to read {result_path}: {e}")
            continue

        # Load hyperparameters.json
        try:
            with open(hyper_path, "r") as file:
                hyperparameters = yaml.safe_load(file)
        except Exception as e:
            print(f"Failed to read {hyper_path}: {e}")
            continue

        # Load metadata.txt
        try:
            with open(meta_path, "r") as f:
                lines = f.readlines()
                model_size = int(lines[1].split(" ")[-2].replace(",", ""))
        except Exception as e:
            print(f"Failed to read {meta_path}: {e}")
            continue

        # Load log_metrics.npy
        try:
            log_metrics = np.load(metric_path)
            average_time = np.mean(log_metrics[:, 1])
            valid_metrics = log_metrics[:, 3][~np.isnan(log_metrics[:, 3])]
            if valid_metrics.size > 0:
                if sort_increasing:
                    val_metric = float(np.max(valid_metrics))
                else:
                    val_metric = float(np.min(valid_metrics))
            else:
                val_metric = float('nan')  # or skip this run
                print(f"  Warning: {metric_path} has only NaNs in column 3")
        except Exception as e:
            print(f"Failed to read {metric_path}: {e}")
            continue

        group_key = make_group_key(hyperparameters, group_keys_to_use)

        groups[group_key].append((test_metric, val_metric, model_size, average_time, dir_path))

    summaries = []

    for group_key, results in groups.items():
        test_scores = [score for score, _, _, _, _ in results]
        val_scores = [score for _, score, _, _, _ in results]
        sizes = [size for _, _, size, _, _ in results]
        times = [time for _, _, _, time, _ in results]
        dirs = [d for _, _, _, _, d in results]

        def compute_mean_std(scores):
            if any(math.isnan(s) for s in scores):
                mean_score = float('nan')
                std_score = float('nan')
            else:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            return mean_score, std_score

        mean_test, std_test = compute_mean_std(test_scores)
        mean_val, std_val = compute_mean_std(val_scores)
        min_test, max_test = min(test_scores), max(test_scores)
        min_val, max_val = min(val_scores), max(val_scores)
        size = sizes[0]  # Model sizes should be the same for constant hyperparams
        time = statistics.mean(times)
        num = len(results)

        subfolders = [d.split('/')[-1] for d in dirs]
        recap = [f"{subfolder}: Val=[{val_score}], Test=[{test_score}]" for subfolder, val_score, test_score in zip(subfolders, val_scores, test_scores)]

        summaries.append({
            "group": group_key,
            "test_mean": mean_test,
            "test_std": std_test,
            "val_mean": mean_val,
            "val_std": std_val,
            "test_min": min_test,
            "test_max": max_test,
            "val_min": min_val,
            "val_max": max_val,
            "model_size": size,
            "avg_time": time,
            "num_runs": num,
            "recap": recap,
        })

    if sort_increasing:
        if sort_by == "mean":
            # summaries.sort(key=lambda x: x["test_mean"] if not math.isnan(x["test_mean"]) else float('inf'))
            summaries.sort(key=lambda x: x["val_mean"] if not math.isnan(x["val_mean"]) else float('inf'))
        elif sort_by == "max":
            # summaries.sort(key=lambda x: x["test_max"] if not math.isnan(x["test"]) else float('inf'))
            summaries.sort(key=lambda x: x["val_max"] if not math.isnan(x["val_max"]) else float('inf'))
        else:
            raise ValueError("`sort_by` should be 'mean' or 'max'")
    else:
        if sort_by == "mean":
            # summaries.sort(key=lambda x: x["test_mean"] if not math.isnan(x["test_mean"]) else float('inf'), reverse=True)
            summaries.sort(key=lambda x: x["val_mean"] if not math.isnan(x["val_mean"]) else float('inf'), reverse=True)
        elif sort_by == "max":
            # summaries.sort(key=lambda x: x["test_min"] if not math.isnan(x["test_min"]) else float('inf'), reverse=True)
            summaries.sort(key=lambda x: x["val_min"] if not math.isnan(x["val_min"]) else float('inf'), reverse=True)
        else:
            raise ValueError("`sort_by` should be 'mean' or 'max'")

    for result in summaries:
        for s in result['recap']:
            print(s)
        print(f"Avg Test: [{result['test_mean']:.6f} ± {result['test_std']:.6f}]")
        print(f"Avg Val: [{result['val_mean']:.6f} ± {result['val_std']:.6f}]")
        print(f"Bound Test: [{result['test_min']:.6f} to {result['test_max']:.6f}]")
        print(f"Bound Val: [{result['val_min']:.6f} to {result['val_max']:.6f}]")
        print(f"{result['group']}")
        print(f"# {result['num_runs']}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Path to specific experiment folder. Should be relative to the damped-linoss home directory (i.e. starts with experiments/)."
    )
    parser.add_argument(
        "-sort_increasing",
        action='store_true',
        help="True: higher values are better. False: lower values are better.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        required=False,
        default="mean",
        help="'mean' or 'max'"
    )
    args = parser.parse_args()

    process_results(
        args.experiment_folder,
        args.sort_increasing,
        args.sort_by,
    )
