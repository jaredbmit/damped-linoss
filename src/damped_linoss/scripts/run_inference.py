"""
This script loads a pre-saved model in .eqx format and computes inference outputs
for specified data. Also provides the option to save model parameter matrices
or hidden states during inference.

Arguments for `run_inference.py`:
`save_model_folder` (str): Relative directory to linoss/ storing model save.
`save_parameters` (bool, optional): If True, saves internal model parameters.
                                    Defaults to False.
`save_states` (bool, optional): If True, saves ssm states during inference
                                (this is data-intensive). Defaults to False.
"""

import time
import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from linoss.models.create_model import create_model
from linoss.data.create_dataset import create_dataset


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def run_inference(
    save_model_folder: str,
    add_noise: float = 0.0,
    data_split: Optional[str] = None,
    save_parameters: Optional[bool] = False,
    save_states: Optional[bool] = False,
):
    """
    Loads a pre-saved model and computes inference outputs on entire dataset.

    Args:
        save_model_folder (str): Relative directory to linoss/ storing model save.
        add_noise (float): Std dev of additive white noise applied to inputs.
            Defaults to zero noise.
        data_split (str): Which split to run inference on, ie "train", "val", or "test".
            Defaults to None, ie inference over all data splits.
        save_parameters (bool, optional): If True, saves internal model parameters.
            Defaults to False.
        save_states (bool, optional): If True, saves ssm states during inference
            (this is data-intensive). Defaults to False.
    """
    save_dir = BASE_DIR / save_model_folder
    output_dir = save_dir / "inference"
    os.makedirs(output_dir, exist_ok=True)
    params_dir = str(save_dir / "params") if save_parameters else None
    states_dir = str(save_dir / "states") if save_states else None

    # Load hyperparameters and create empty model
    with open(save_dir / "hyperparameters.pkl", "rb") as f:
        hyperparameters = pickle.load(f)
    empty_model, empty_state = create_model(**hyperparameters)

    # Load model
    loaded_model = eqx.tree_deserialise_leaves(save_dir / "model.eqx", empty_model)
    loaded_state = eqx.tree_deserialise_leaves(save_dir / "state.eqx", empty_state)
    inference_model = eqx.tree_inference(loaded_model, True)  # Inference mode

    # Save model parameters, if requested
    if save_parameters:
        inference_model.save_params(params_dir)

    # Load dataset arguments and create dataset
    with open(save_dir / "dataset_args.pkl", "rb") as f:
        dataset_args = pickle.load(f)
    dataset_args["data_dir"] = str(BASE_DIR / "data")
    dataset = create_dataset(**dataset_args)

    @eqx.filter_jit
    def evaluate(model, x, s, key):
        if model.nondeterministic and model.stateful:
            out, _ = jax.vmap(model, in_axes=(0, None, None))(x, s, key)
        elif model.stateful:
            out, _ = jax.vmap(model, in_axes=(0, None))(x, s)
        elif model.nondeterministic:
            out = jax.vmap(model, in_axes=(0, None))(x, key)
        else:
            out = jax.vmap(model)(x)
        return out

    @eqx.filter_jit
    def evaluate_transformer(model, x):
        out = jax.vmap(model.autoregressive_inference)(x)
        return out

    # Run inference by split
    for split, dataloader in dataset.dataloaders.items():
        if data_split and split != data_split:
            continue

        print(f"Running inference on split {split}")
        inputs = jnp.array(dataloader.data)
        truth = jnp.array(dataloader.labels)

        # Add noise, if specified
        noise = np.random.normal(scale=add_noise, size=inputs.shape)
        inputs += noise

        # Time inference
        start_time = time.time()

        if hyperparameters["model_name"] == "Transformer":
            outputs = evaluate_transformer(inference_model, inputs)
        elif states_dir is None:
            outputs = evaluate(
                inference_model, inputs, loaded_state, hyperparameters["key"]
            )
        else:
            outputs = []
            for x in tqdm(inputs):
                output, _ = inference_model(
                    x, loaded_state, hyperparameters["key"], save_dir=states_dir
                )
                outputs.append(output)
            outputs = jnp.stack(outputs)

        outputs = jnp.squeeze(outputs)
        inputs = jnp.squeeze(inputs)

        jax.block_until_ready(outputs)  # Ensure timing includes compute
        end_time = time.time()
        total_time = end_time - start_time

        print(f"Inference time: {total_time:.4f} seconds")
        print(f"Average time per sample: {total_time / len(inputs):.6f} seconds")
        print(f"MSE: {np.mean((outputs - truth)**2)}")

        # Only save original data
        if dataset_args["time_duration"] is not None:
            inputs = inputs[..., 1:]

        # Save inputs and outputs
        print(f"Saving inference inputs and outputs to {output_dir}")
        save_pickle(output_dir / f"inputs_{split}.pkl", np.array(inputs))
        save_pickle(output_dir / f"outputs_{split}.pkl", np.array(outputs))
        save_pickle(output_dir / f"truth_{split}.pkl", np.array(truth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_folder",
        type=str,
        required=True,
        help="Path to model save folder, relative to base directory linoss/",
    )
    parser.add_argument(
        "--add_noise",
        type=float,
        required=False,
        default=0.0,
        help="Specifies standard deviation of additive white noise to inputs, \
            Defaults to zero noise.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        required=False,
        default=None,
        help="Which split of dataset to run inference on. Defaults to None, \
            ie inference over all data splits.",
    )
    parser.add_argument(
        "--save_parameters",
        action="store_true",
        help="Whether or not to save model parameters",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Whether or not to save internal ssm states during inference",
    )
    args = parser.parse_args()

    run_inference(
        args.model_save_folder,
        add_noise=args.add_noise,
        data_split=args.data_split,
        save_parameters=args.save_parameters,
        save_states=args.save_states,
    )
