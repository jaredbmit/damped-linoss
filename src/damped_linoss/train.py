"""
This module defines functions for creating datasets, building models,
and training them using JAX and Equinox.

The main function, `create_dataset_model_and_train`, is designed to initialize the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:
- `run_folder`: Absolute path to training run folder.
- `hyperparameters`: Dictionary of model hyperparameters.

The module also includes the following key functions:
- `calc_output`: Computes the model output, handling stateful and nondeterministic
                 models with JAX's `vmap` for batching.
- `classification_loss`: Computes the loss for classification tasks, including
                         optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional
                     regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters
               based on the computed gradients.
- `evaluate`: Computes the classification/non-classification metric given a 
              specific dataloader split.
- `train_model`: Handles the training loop, managing metrics, early stopping,
                 and saving progress at regular intervals.
"""
import os
import time
import warnings
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from jaxtyping import PyTree

from damped_linoss.data.create_dataset import create_dataset
from damped_linoss.models.create_model import create_model
from damped_linoss.models.LinOSS import LinOSS
from damped_linoss.models.LRU import LRU
from damped_linoss.models.S5 import S5

# Ignore warning in loss fn calculation
warnings.simplefilter("ignore", category=jnp.ComplexWarning)


def count_params(model):
    leaves, _ = jax.tree_util.tree_flatten(model)
    param_leaves = [leaf for leaf in leaves if isinstance(leaf, jnp.ndarray)]
    num_params = sum(leaf.size for leaf in param_leaves)
    num_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in param_leaves)
    return num_params, num_bytes


def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    bsz, _, _ = X.shape
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(
                model,
                axis_name="batch",
                in_axes=(0, None, None),
                out_axes=(0, None),
            )(X, state, key)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)), state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(model, X, y, state, key):
    pred_y, state = calc_output(model, X, state, key, model.stateful, model.nondeterministic)
    return jnp.mean((jnp.squeeze(pred_y) - jnp.squeeze(y)) ** 2.0), state


@eqx.filter_jit
def make_step(model, X, y, loss_fn, state, opt, opt_state, key):
    (value, state), grads = loss_fn(model, X, y, state, key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, value


def evaluate(inference_model, state, dataloader_iter, key):
    predictions = []
    labels = []
    for data in dataloader_iter:
        eval_key, key = jr.split(key, 2)
        X, y = data
        prediction, _ = calc_output(
            inference_model,
            X,
            state,
            eval_key,
            inference_model.stateful,
            inference_model.nondeterministic,
        )
        predictions.append(prediction)
        labels.append(y)
    prediction = jnp.vstack(predictions)
    y = jnp.vstack(labels)

    if inference_model.classification:
        metric = jnp.mean(jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1))
    else:
        metric = jnp.mean((jnp.squeeze(prediction) - jnp.squeeze(y)) ** 2.0)
    
    return metric


def train_model(
    run_folder: str,
    model,
    state,
    dataset,
    classification: bool,
    num_steps: int,
    print_steps: int,
    batch_size: int,
    lr: float,
    key: jax.Array,
):
    # Initialize model optimizer
    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Model saving
    checkpoint_folder = os.path.join(run_folder, "checkpoints")
    def copy_tree(tree, temp_file):
        eqx.tree_serialise_leaves(temp_file, tree)
        return eqx.tree_deserialise_leaves(temp_file, tree)
    best_model = copy_tree(model, os.path.join(checkpoint_folder, f"model_0.eqx"))
    best_state = copy_tree(state, os.path.join(checkpoint_folder, f"state_0.eqx"))

    # Classification vs. Regression
    if classification:
        improvement = lambda x, y: x > y
        loss_fn = classification_loss
        best_val_metric = -jnp.inf
    else:
        improvement = lambda x, y: x < y
        loss_fn = regression_loss
        best_val_metric = jnp.inf

    print("Starting training.")
    running_losses = []
    val_metrics = []
    step_times = []
    running_loss = 0.0
    counter = 0
    start = time.time()
    for step, data in zip(
        range(num_steps),
        dataset.dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        # Make step
        X, y = data
        step_key, key = jr.split(key, 2)
        model, state, opt_state, value = make_step(model, X, y, loss_fn, state, opt, opt_state, step_key)
        running_loss += value

        # Evaluation @ print_step
        if (step + 1) % print_steps == 0:
            end = time.time()
            total_time = end - start

            # Validation metrics
            val_key, key = jr.split(key, 2)
            inference_model = eqx.tree_inference(model, value=True)
            val_iter = dataset.dataloaders["val"].loop_epoch(batch_size, val_key)
            val_metric = evaluate(inference_model, state, val_iter, val_key)
            print(
                f"Step: {step + 1}, "
                f"Loss: {running_loss / print_steps}, "
                f"Validation metric: {val_metric}, "
                f"Time: {total_time}"
            )
            running_losses.append(running_loss / print_steps)
            val_metrics.append(val_metric)
            step_times.append(total_time)
            running_loss = 0.0

            # Improvement checking / early stopping
            if improvement(val_metric, best_val_metric):
                counter = 0
                best_val_metric = val_metric
                best_model = copy_tree(model, os.path.join(checkpoint_folder, f"model_{step+1}.eqx"))
                best_state = copy_tree(state, os.path.join(checkpoint_folder, f"state_{step+1}.eqx"))
            else:
                counter += 1
                if counter >= 10000:  # temp
                    print("--- Early Stopping. ---")
                    break

            start = time.time()

    # Compute test metric
    test_key, key = jr.split(key, 2)
    best_inference_model = eqx.tree_inference(best_model, value=True)
    test_iter = dataset.dataloaders["test"].loop_epoch(batch_size, test_key)
    test_metric = evaluate(best_inference_model, best_state, test_iter, test_key)

    print(f"Test metric: {test_metric}")
    with open(os.path.join(run_folder, "test_metric.txt"), "w") as f:
        f.write(str(test_metric))

    # Log final results
    log_data = jnp.stack([
        jnp.arange(print_steps, (len(val_metrics) + 1) * print_steps, print_steps),
        jnp.array(step_times),
        jnp.array(running_losses),
        jnp.array(val_metrics)
    ], axis=1)
    jnp.save(os.path.join(run_folder, "log_metrics.npy"), log_data)

    return best_model, best_state


def create_dataset_model_and_train(
    run_folder: str,
    hyperparameters: dict,
):
    seed = safe_load(hyperparameters, "seed", int)
    model_name = safe_load(hyperparameters, "model_name", str)
    dataset_name = safe_load(hyperparameters, "dataset_name", str)
    dataset_key, model_key, train_key = jr.split(jr.PRNGKey(seed), 3)

    def delete_file_if_exists(file):
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted: {file}")

    delete_file_if_exists(os.path.join(run_folder, "metadata.txt"))
    delete_file_if_exists(os.path.join(run_folder, "test_metric.txt"))
    delete_file_if_exists(os.path.join(run_folder, "log_metrics.npy"))
    delete_file_if_exists(os.path.join(run_folder, "model.eqx"))
    delete_file_if_exists(os.path.join(run_folder, "state.eqx"))

    os.makedirs(os.path.join(run_folder, "checkpoints"), exist_ok=True)

    print(f"Creating dataset {dataset_name}")
    dataset = create_dataset(
        name=dataset_name,
        data_dir=safe_load(hyperparameters, "data_dir", str),
        classification=safe_load(hyperparameters, "classification", bool),
        time_duration=safe_load(hyperparameters, "time_duration", float) if safe_load(hyperparameters, "include_time", bool) else None,
        use_presplit=safe_load(hyperparameters, "use_presplit", bool),
        key=dataset_key
    )

    print(f"Creating model {model_name}")
    hyperparameters |= {"input_dim": dataset.data_dim, "output_dim": dataset.label_dim}
    model, state = create_model(
        hyperparameters=hyperparameters,
        key=model_key,
    )
    
    # Report metadata
    with open(os.path.join(run_folder, "metadata.txt"), "w") as f:
        n_params, n_bytes = count_params(model)
        f.write(f"Experiment conducted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
        f.write(f"# of Parameters: {n_params:,} \n")
        f.write(f"Memory: {n_bytes/1024/1024:.2f} MiB")

    # Train
    model, state = train_model(
        run_folder,
        model,
        state,
        dataset,
        classification=safe_load(hyperparameters, "classification", bool),
        num_steps=safe_load(hyperparameters, "num_steps", int),
        print_steps=safe_load(hyperparameters, "print_steps", int),
        batch_size=safe_load(hyperparameters, "batch_size", int),
        lr=safe_load(hyperparameters, "lr", float),
        key=train_key,
    )

    return model, state
