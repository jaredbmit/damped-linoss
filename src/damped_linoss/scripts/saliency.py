"""
This script approximates weight saliency for recurrent matrix parameters in SSM layers.

Weight saliency is defined as in the OBD, OBS papers
https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf
https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf

General formula for weight saliency of parameter w_i is 1/2 (w_i^2 / (H^-1)_ii).
This is intensive to compute since Hessian is large and inverting is expensive.
Instead we use the simplifying assumption (H^-1)_ii ≈ (H_ii)^-1.
Hessian diagonal elements are computed via Hessian vector products instead of constructing the full Hessian matrix.
    More info: https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html

Usage:
    uv run python -m damped_linoss.scripts.saliency --run_folder experiments/<path_to_run_folder>
"""
import os
import re
import yaml
import glob
import tqdm
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model
from damped_linoss.train import calc_output


#############################
# Section: Helper functions #
#############################

def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


def count_nontrivial_leaves(tree):
        leaves, _ = jax.tree.flatten(tree)
        return sum(x is not None for x in leaves)


def hvp(f, primals, tangents):
    """
    Hessian vector product of "f" evaluated at "primals" with "tangents", i.e. grad(f)(x)v.
    https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode

    Arguments:
        f: function from weights W to loss value
        x: current set of weights to linearize Hessian at
        v: perturbation vector
    
    Returns:
        Hv
    """
    # return jax.grad(lambda x: jnp.vdot(jax.grad(f)(x), v))(x)  # More intuitive but less efficient
    return jax.jvp(jax.grad(f), primals, tangents)[1]


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


###########################
# Section: Loss functions #
###########################

def classification_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)), state


def regression_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean((jnp.squeeze(pred_y) - jnp.squeeze(y)) ** 2.0), state


########
# Main #
########

def compute_saliency(run_folder):
    ## ------ Load model configuration ------ ##

    # Load run hyperparameters
    hyperparameters_path = os.path.join(run_folder, "hyperparameters.yaml")
    with open(hyperparameters_path, "r") as f:
        hyperparameters = yaml.safe_load(f)

    seed = safe_load(hyperparameters, "seed", int)
    dataset_key, model_key, key = jr.split(jr.PRNGKey(seed), 3)

    # Load dataset
    dataset = create_dataset(
        name=safe_load(hyperparameters, "dataset_name", str),
        data_dir=safe_load(hyperparameters, "data_dir", str),
        classification=safe_load(hyperparameters, "classification", bool),
        time_duration=safe_load(hyperparameters, "time_duration", float) if safe_load(hyperparameters, "include_time", bool) else None,
        use_presplit=safe_load(hyperparameters, "use_presplit", bool),
        key=dataset_key,
    )
    dataloader = dataset.dataloaders["val"]

    # Create empty model
    hyperparameters |= {"input_dim": dataset.data_dim, "output_dim": dataset.label_dim}
    empty_model, empty_state = create_model(
        hyperparameters=hyperparameters,
        key=model_key,
    )

    # Load best model instance
    checkpoint_folder = os.path.join(run_folder, "checkpoints")
    pattern = os.path.join(checkpoint_folder, "model_*.eqx")
    results = glob.glob(pattern, recursive=True)
    best_idx = 0
    for r in results:
        idx = int(re.search(r"model_(\d+)", r).group(1))
        if idx > best_idx:
            best_idx = idx
    model_path = os.path.join(checkpoint_folder, f"model_{best_idx}.eqx")
    state_path = os.path.join(checkpoint_folder, f"state_{best_idx}.eqx")
    model = eqx.tree_deserialise_leaves(model_path, empty_model)
    state = eqx.tree_deserialise_leaves(state_path, empty_state)
    inference_model = eqx.tree_inference(model, value=True)  # Inference mode

    ## ------ Compute Hessian ------ ##

    # Filter damping parameters
    keys_to_check = ["A_diag", "G_diag", "dt"]
    def filter_spec(path, x):
        return getattr(path[-1], "name", None) in keys_to_check
    filtered_tree = jax.tree.map_with_path(filter_spec, inference_model)
    diff_tree, static_tree = eqx.partition(inference_model, filtered_tree)
    
    # Flatten tree
    params, treedef = jax.tree.flatten(diff_tree)
    shapes = [p.shape for p in params]
    sizes = [p.size for p in params]
    flat_params = jnp.concatenate([p.ravel() for p in params])
    
    def unflatten_params(flat):
        leaves = []
        idx = 0
        for shape, size in zip(shapes, sizes):
            leaf = flat[idx:idx+size].reshape(shape)
            leaves.append(leaf)
            idx += size
        return jax.tree.unflatten(treedef, leaves)

    # Loss function
    if safe_load(hyperparameters, "classification", bool):
        loss = classification_loss
    else:
        loss = regression_loss
    
    def loss_flat(flat_params, X, y, key):
        diff_tree = unflatten_params(flat_params)
        model = eqx.combine(diff_tree, static_tree)
        value, _ = loss(model, X, y, state, key)
        return value
    
    # Jit / batching
    @jax.jit
    def hvp_jit(flat, v, X, y, key):
        return hvp(lambda w: loss_flat(w, X, y, key), (flat,), (v,))
        
    def hvp_batch(flat, V_chunk, X, y, key):
        return jax.vmap(lambda v: hvp_jit(flat, v, X, y, key))(V_chunk)

    # Approximate diagonal Hessian
    chunk_size = 32
    batch_size = 32
    num_batches = dataloader.size // batch_size  # Ignore batching remainder
    data_iter = dataloader.loop_epoch(batch_size)
    P = flat_params.size
    H_diag = jnp.zeros(P)
    for i in tqdm.tqdm(range(num_batches)):
        step_key, key = jr.split(key, 2)
        X, y = next(data_iter)

        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)

            # (chunk_size, P) -> (chunk_size, P)
            V_chunk = jnp.eye(P, dtype=flat_params.dtype)[start:end]
            Hv_chunk = hvp_batch(flat_params, V_chunk, X, y, step_key)

            # Hv_chunk[k, :] = H * e_(start+k)
            diag_chunk = jnp.diag(Hv_chunk[:, start:end])

            # Accumulate
            H_diag = H_diag.at[start:end].add(diag_chunk)

    H_diag = H_diag / float(num_batches)
    saliency = 0.5 * flat_params**2 * H_diag
    saliency_tree = unflatten_params(saliency)

    ## ------ Report results ------ ##

    # Saliency statistics
    stats_text = f"""
    SALIENCY STATISTICS
    {'='*40}
    Total Parameters:           {len(saliency):,}
    Total Saliency:             {np.sum(saliency):.4e}
    Mean Saliency:              {np.mean(saliency):.4e}
    Median Saliency:            {np.median(saliency):.4e}
    Std Saliency:               {np.std(saliency):.4e}
    Max Saliency:               {np.max(saliency):.4e}
    Min Saliency:               {np.min(saliency):.4e}
    """
    print(stats_text)

    figures_folder = os.path.join(run_folder, "figures")
    os.makedirs(figures_folder, exist_ok=True)

    # Saliency histogram
    fig1, ax1 = plt.subplots(1, 1, figsize=(11, 7))
    saliency_nonzero = saliency[saliency > 0]
    ax1.hist(np.log10(saliency_nonzero + 1e-10), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Log10(Saliency)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Saliency Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    fig1.savefig(os.path.join(figures_folder, "saliency_histogram.png"), dpi=300)

    # Saliency vs. weight
    fig2, ax2 = plt.subplots(1, 1, figsize=(11, 7))
    weight_mag = np.abs(flat_params)
    ax2.scatter(weight_mag, saliency, s=50, marker='o', alpha=0.7, edgecolors='k', linewidth=0.8)
    ax2.set_xlabel('|Weight|', fontsize=11)
    ax2.set_ylabel('Saliency', fontsize=11)
    ax2.set_title('Saliency vs Weight Magnitude', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    fig2.savefig(os.path.join(figures_folder, "saliency_vs_weight.png"), dpi=300)

    # Saliency vs. eigenvalues
    print("Computing eigenvalues...")
    eigenvalues = []
    saliencies = []
    for block, saliency_block in zip(inference_model.blocks, saliency_tree.blocks):
        # model parameters
        layer = block.layer
        A = layer.A_diag
        G = layer.G_diag
        dt = layer.dt

        # tree mirror containing saliency info
        # aggregate across parameters via summing saliencies
        saliency_layer = saliency_block.layer
        saliency_values = saliency_layer.G_diag + saliency_layer.A_diag + saliency_layer.dt

        # soft-project exactly as the model does
        A_proj, G_proj, dt_proj = layer._soft_project_AGdt(A, G, dt)

        # collect complex eigenvalues for this layer
        eigenvalues_this_layer = []
        for a, g, d in zip(A_proj, G_proj, dt_proj):
            M_sym = recurrent_matrix(a.item(), g.item(), d.item())
            evals = list(M_sym.eigenvals().keys())
            evals_np = [complex(e.evalf()) for e in evals]
            eigenvalues_this_layer.extend(evals_np)

        # duplicate saliency values to match 2x number of eigenvalues
        saliency_values = [x for s in saliency_values for x in (s, s)]
        
        eigenvalues.append(np.array(eigenvalues_this_layer, dtype=np.complex128))
        saliencies.append(np.array(saliency_values))

    print("Plotting eigenvalues.")
    fig3, ax3 = plt.subplots(1, len(eigenvalues), figsize=(11, 4))
    vmin, vmax = -5, 0  # log scale
    for i, (e, s) in enumerate(zip(eigenvalues, saliencies)):
        s_log = np.log10(np.abs(s) + 1e-10)

        scatter = ax3[i].scatter(e.real, e.imag, c=s_log, cmap='viridis', s=100, marker='o', alpha=0.7,
                    edgecolors='k', linewidth=0.8, vmin=vmin, vmax=vmax)
        
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax3[i].plot(np.cos(theta), np.sin(theta), 'k--', lw=1.2)

        # Formatting
        ax3[i].axhline(0, color='k', lw=0.5)
        ax3[i].axvline(0, color='k', lw=0.5)
        ax3[i].axis('equal')
        ax3[i].set_xlim(-1.1, 1.1)
        ax3[i].grid(True, ls=":", alpha=0.5)
        ax3[i].set_xlabel("Real part", fontsize=13)
        ax3[i].set_title(f"Eigenvalues for layer {i}", fontsize=13)
        
    ax3[0].set_ylabel("Imaginary part", fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax3[-1], pad=0.02)
    cbar.set_label('Log10(Saliency)', fontsize=13, rotation=270, labelpad=20)
    fig3.savefig(os.path.join(figures_folder, "saliency_eigenvalues.png"), dpi=300)

    # Saliency vs. eigenvalue magnitude
    fig4, ax4 = plt.subplots(1, 1, figsize=(11, 7))
    all_eigs = []
    all_sals = []
    for i, (e, s) in enumerate(zip(eigenvalues, saliencies)):
        abs_e = np.abs(e)
        all_eigs.append(abs_e)
        all_sals.append(s)
        ax4.scatter(
            abs_e, s,
            s=110, marker='o', alpha=0.75,
            edgecolors='k', linewidth=1.0,
            label=f"Layer {i}"
        )
    all_eigs = np.array(all_eigs).flatten()
    all_sals = np.array(all_sals).flatten()
    log_sals = np.log10(np.abs(all_sals) + 1e-12)

    # Best-fit line
    a, b = np.polyfit(all_eigs, log_sals, 1)
    x_line = np.linspace(all_eigs.min(), all_eigs.max(), 200)
    y_line = 10 ** (a * x_line + b)
    ax4.plot(x_line, y_line, color="black", linestyle='--', linewidth=2.5)

    # Axis labels and title
    ax4.set_xlabel(r'$|\lambda|$', fontsize=18)
    ax4.set_ylabel(r'$\log_{10}(\text{Saliency})$', fontsize=18)
    ax4.set_title('Saliency vs Eigenvalue Magnitude', fontsize=20, fontweight='bold')
    ax4.tick_params(axis='both', labelsize=15)
    ax4.set_yscale('log')
    ax4.set_ylim([0.5e-7, 2e4])
    ax4.grid(alpha=0.3)
    ax4.legend(loc='upper left', fontsize=14, framealpha=0.9)

    # Text box positioned directly below the legend
    textbox = (
        r'Best Fit: $\log_{10}(s) = '
        f'{a:.3f} |\lambda| + {b:.3f}$'
    )
    ax4.text(
        0.02,
        0.76,
        textbox,
        transform=ax4.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    fig4.savefig(os.path.join(figures_folder, "saliency_vs_magnitude.png"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder",
        type=str,
        required=True,
        help="Path to specific run folder. Should be relative to the damped-linoss home directory (i.e. starts with experiments/)."
    )
    args = parser.parse_args()

    compute_saliency(
        args.run_folder
    )
