import numpy as np
import pickle
import os
from copy import deepcopy
from pathlib import Path

# linoss/ directory
BASE_DIR = Path(__file__).resolve().parent.parent


def simulate_dynamics(A, B, C, D, input, x0):
    x_prev = x0
    outputs = []
    for u in input:
        x_next = A * x_prev + B * u
        y_next = C * x_prev + D * u
        x_prev = x_next
        outputs.append(deepcopy(y_next))

    return outputs


if __name__ == "__main__":
    num_samples = 100
    num_timesteps = 1000
    out_dir = BASE_DIR / "data" / "processed" / "synthetic_regression"
    os.makedirs(out_dir, exist_ok=True)

    x0 = 0
    A = 0.8
    B = 1
    C = 1
    D = 0

    inputs = []
    outputs = []
    for n in range(num_samples):
        signal = np.random.normal(size=(num_timesteps,))
        output = simulate_dynamics(A, B, C, D, signal, x0)
        inputs.append(signal)
        outputs.append(output)

    inputs = np.array(inputs)[..., np.newaxis]
    outputs = np.array(outputs)

    with open(out_dir / "X_train.pkl", "wb") as f:
        pickle.dump(inputs[0:70], f)
    with open(out_dir / "y_train.pkl", "wb") as f:
        pickle.dump(outputs[0:70], f)
    with open(out_dir / "X_val.pkl", "wb") as f:
        pickle.dump(inputs[70:85], f)
    with open(out_dir / "y_val.pkl", "wb") as f:
        pickle.dump(outputs[70:85], f)
    with open(out_dir / "X_test.pkl", "wb") as f:
        pickle.dump(inputs[85:100], f)
    with open(out_dir / "y_test.pkl", "wb") as f:
        pickle.dump(outputs[85:100], f)
