import os
import yaml
import glob
import numpy as np
import math
import statistics
from collections import defaultdict
from sys import argv


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
    

def main(exp_root):
    groups = defaultdict(list)
    group_keys_to_use = ["model_name", "dataset_name", "lr", "state_dim", "hidden_dim", "num_blocks", "include_time", "weight_decay", "cosine_annealing", "batch_size", "r_min", "r_max", "theta_min", "theta_max", "A_min", "A_max", "G_min", "G_max", "dt_std", "drop_rate"]

    # Find all results.txt in run_XXX folders under exp_root
    pattern = os.path.join(exp_root, "run_*/test_metric.txt")
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
                val_metric = np.min(valid_metrics)
            else:
                val_metric = float('nan')  # or skip this run
                print(f"  Warning: {metric_path} has only NaNs in column 3")
        except Exception as e:
            print(f"Failed to read {metric_path}: {e}")
            continue

        group_key = make_group_key(hyperparameters, group_keys_to_use)

        groups[group_key].append((test_metric, val_metric, model_size, average_time))

    summaries = []

    for group_key, results in groups.items():
        test_scores = [score for score, _, _, _ in results]
        val_scores = [score for _, score, _, _ in results]
        sizes = [size for _, _, size, _ in results]
        times = [time for _, _, _, time in results]

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
        size = sizes[0]  # Model sizes should be the same for constant hyperparams
        time = statistics.mean(times)
        num = len(results)

        summaries.append({
            "group": group_key,
            "test_mean": mean_test,
            "test_std": std_test,
            "val_mean": mean_val,
            "val_std": std_val,
            "model_size": size,
            "avg_time": time,
            "num_runs": num,
        })

    summaries.sort(key=lambda x: x["val_mean"] if not math.isnan(x["val_mean"]) else float('inf'), reverse=True)
    # summaries.sort(key=lambda x: x["test_mean"], reverse=True)

    for result in summaries:
        print(f"Test: [{result['test_mean']:.6f} ± {result['test_std']:.6f}]")
        print(f"Val: [{result['val_mean']:.6f} ± {result['val_std']:.6f}]")
        print(f"{result['group']}")
        print(f"# {result['num_runs']}")
        print()

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python process_results.py <experiment_root_folder>")
    else:
        main(argv[1])
