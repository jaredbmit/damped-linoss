"""
Scrapes an experiment run folder in experiments/{model_name}/{datset_name}/run_{id}/
Loads all model checkpoints, evaluates recurrence A matrix eigenvalues, plots evolution over time

Usage:
    uv run python -m damped_linoss.scripts.analyze_eigenvalues --run_folder experiments/<path_to_run_folder>

Saves figure in the run folder under figures/
"""
import re
import os
import glob
import yaml
import numpy as np
import sympy as sp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import argparse

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model


def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


def recurrent_matrix(a, g, dt):
    """
    Symbolic expression for the 2×2 recurrent block M_i.
    """
    M = sp.Matrix(
        [
            [1 / (1 + dt * g), -a * dt / (1 + dt * g)],
            [dt / (1 + dt * g), 1 - a * dt**2 / (1 + dt * g)],
        ]
    )
    return M


def analyze_eigenvalues(run_folder: str):
    # ------------------------------------------------------------------ #
    # 1. Load hyperparameters and build the *base* model PyTree
    # ------------------------------------------------------------------ #
    print("Creating base model.")
    hyperparameters_path = os.path.join(run_folder, "hyperparameters.yaml")
    with open(hyperparameters_path, "r") as f:
        hyperparameters = yaml.safe_load(f)

    dataset = create_dataset(
        name=safe_load(hyperparameters, "dataset_name", str),
        data_dir=safe_load(hyperparameters, "data_dir", str),
        classification=safe_load(hyperparameters, "classification", bool),
        time_duration=safe_load(hyperparameters, "time_duration", float) if safe_load(hyperparameters, "include_time", bool) else None,
        use_presplit=safe_load(hyperparameters, "use_presplit", bool),
        key=jr.PRNGKey(0),  # Seed irrelevant here
    )

    hyperparameters |= {"input_dim": dataset.data_dim, "output_dim": dataset.label_dim}
    model, state = create_model(
        hyperparameters=hyperparameters,
        key=jr.PRNGKey(0),  # Seed irrelevant here
    )

    # ------------------------------------------------------------------ #
    # 2. Find all checkpoint files
    # ------------------------------------------------------------------ #
    checkpoints_folder = os.path.join(run_folder, "checkpoints")
    checkpoint_pattern = os.path.join(checkpoints_folder, "model_*.eqx")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_folder}")
    
    # ------------------------------------------------------------------ #
    # 3. Loop over checkpoints – collect per-layer complex eigenvalues
    # ------------------------------------------------------------------ #
    all_eigs_per_layer = []   # list of list of np.ndarray[complex] : [checkpoint][layer]
    steps = []

    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"Loading model {i+1} out of {len(checkpoint_files)}")

        # extract step number from filename
        base = os.path.basename(checkpoint_file)
        step_str = base.replace("model_", "").replace(".eqx", "")
        step = int(step_str)
        steps.append(step)

        # load the checkpoint into the *base* model tree
        model_checkpoint = eqx.tree_deserialise_leaves(checkpoint_file, model)

        # collect eigenvalues from every recurrent block
        layer_eigs = []   # list of np.ndarray[complex] (one entry per layer)

        print("Computing eigenvalues.")
        for block in model_checkpoint.blocks:
            layer = block.layer
            A = layer.A_diag
            G = layer.G_diag
            dt = layer.dt

            # soft-project exactly as the model does
            A_proj, G_proj, dt_proj = layer._soft_project_AGdt(A, G, dt)

            # collect complex eigenvalues for this layer
            evals_this_layer = []
            for a, g, d in zip(A_proj, G_proj, dt_proj):
                M_sym = recurrent_matrix(a.item(), g.item(), d.item())
                evals = list(M_sym.eigenvals().keys())
                evals_np = [complex(e.evalf()) for e in evals]
                evals_this_layer.extend(evals_np)

            layer_eigs.append(np.array(evals_this_layer, dtype=np.complex128))

        all_eigs_per_layer.append(layer_eigs)   # checkpoint → [layer0, layer1, ...]

    # ------------------------------------------------------------------ #
    # 4. Plot eigenvalue trajectories: start → trace → end (per layer)
    # ------------------------------------------------------------------ #
    print("Plotting eigenvalues.")
    os.makedirs(os.path.join(run_folder, "figures"), exist_ok=True)
    fig_path = os.path.join(run_folder, "figures", "eigenvalue_trajectories.png")

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    handles = []

    # Track trajectories across steps: [layer][eig_idx][checkpoint #]
    n_eigs = len(all_eigs_per_layer[0][0])
    n_layers = len(all_eigs_per_layer[0])
    trajectories = [[[] for _ in range(n_eigs)] for _ in range(n_layers)]

    # Fill trajectories
    print("Flipping eigenvalue tree.")
    for layer_eigs_list in all_eigs_per_layer:
        for layer_idx, eigs in enumerate(layer_eigs_list):
            for eig_idx, eig in enumerate(eigs):
                trajectories[layer_idx][eig_idx].append(eig)

    # Plot trace (low alpha), start (circle), end (star)
    for layer_idx in range(n_layers):
        print(f"Plotting trajectories for layer {layer_idx+1} out of {n_layers}.")
        color = cmap(layer_idx % cmap.N)
        layer_trajs = trajectories[layer_idx]

        for traj in layer_trajs:
            traj = np.array(traj)

            # Trace line (intermediate points, low alpha)
            plt.plot(traj.real, traj.imag, color=color, alpha=0.25, linewidth=1.2)

            # # Start point
            # plt.scatter(traj[0].real, traj[0].imag,
            #             color=color, s=50, marker='o', edgecolors='k', linewidth=0.8, zorder=5)

            # End point
            plt.scatter(traj[-1].real, traj[-1].imag,
                        color=color, s=70, marker='o', edgecolors='k', linewidth=0.8, zorder=5)

        # Legend: only once per layer
        handles.append(plt.Line2D([], [], color=color, marker='o', markersize=8,
                                markeredgecolor='k', linestyle='None', label=f"Layer {layer_idx} (end)"))
        # handles.append(plt.Line2D([], [], color=color, marker='*', markersize=10,
        #                         markeredgecolor='k', linestyle='None', label=f"Layer {layer_idx} (end)"))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.2, label="Unit circle")

    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("Eigenvalue Trajectories During Training\n(end: circle, trace: faint line)")
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True, ls=":", alpha=0.5)

    # Custom legend
    handles.append(plt.Line2D([], [], color='k', ls='--', lw=1.2, label="Unit circle"))
    plt.legend(handles=handles, loc="upper right", fontsize='small', ncol=2)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Trajectory figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_folder",
        type=str,
        required=True,
        help="Path to specific experiment folder. Should be relative to the damped-linoss home directory (i.e. starts with experiments/)."
    )
    parser.add_argument(
        "--run_ids",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="ID numbers of runs within experiment folder to use. If not specified, defaults to all IDs found in the experiment folder."
    )
    args = parser.parse_args()

    # Find all results.txt in run_XXX folders under experiment_folder
    pattern = os.path.join(args.experiment_folder, "run_*/test_metric.txt")
    results = glob.glob(pattern, recursive=True)

    # Downsampled to only specified run_ids
    results_subset = []
    if args.run_ids:
        for r in results:
            m = re.search(r"run_(\d+)", r)
            if m and int(m.group(1)) in args.run_ids:
                results_subset.append(r)
        results = results_subset
    
    # Compute all eigenvalue trajectories
    for i, result_path in enumerate(results):
        print(f"Evaluating run {i+1}/{len(results)}.")
        dir_path = os.path.dirname(result_path)
        analyze_eigenvalues(dir_path)
