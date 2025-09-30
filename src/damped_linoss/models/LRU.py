"""
Code modified from https://gist.github.com/Ryu1845/7e78da4baa8925b4de482969befa949d

This module implements the `LRU` class, a model architecture using JAX and Equinox.

Attributes of the `LRU` class:
- `linear_encoder`: The linear encoder applied to the input time series data.
- `blocks`: A list of `LRUBlock` instances, each containing the LRU layer, normalization, GLU, and dropout.
- `linear_decoder`: The final linear layer that outputs the model predictions.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes the following classes and functions:
- `LRULayer`: A single LRU layer that applies complex-valued transformations and projections to the input.
- `LRUBlock`: A block consisting of normalization, LRU layer, GLU, and dropout, used as a building block for the `LRU`
              model.
- `binary_operator_diag`: A helper function used in the associative scan operation within `LRULayer` to process diagonal
                          elements.
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from damped_linoss.models.common import GLU


def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class LRULayer(eqx.Module):
    nu_log: jnp.ndarray
    theta_log: jnp.ndarray
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray
    gamma_log: jnp.ndarray

    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, *, key):
        u1_key, u2_key, B_re_key, B_im_key, C_re_key, C_im_key, D_key = jr.split(key, 7)

        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, theta_max].
        u1 = jr.uniform(u1_key, shape=(state_dim,))
        u2 = jr.uniform(u2_key, shape=(state_dim,))
        self.nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        self.theta_log = jnp.log(theta_max * u2)

        # Glorot initialized Input/Output projection matrices
        self.B_re = jr.normal(B_re_key, shape=(state_dim, hidden_dim)) / jnp.sqrt(2 * hidden_dim)
        self.B_im = jr.normal(B_im_key, shape=(state_dim, hidden_dim)) / jnp.sqrt(2 * hidden_dim)
        self.C_re = jr.normal(C_re_key, shape=(hidden_dim, state_dim)) / jnp.sqrt(state_dim)
        self.C_im = jr.normal(C_im_key, shape=(hidden_dim, state_dim)) / jnp.sqrt(state_dim)
        self.D = jr.normal(D_key, shape=(hidden_dim,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    def __call__(self, x):
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im
        # Running the LRU + output projection
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.associative_scan(
            binary_operator_diag, elements
        )  # all x_k
        y = jax.vmap(lambda z, u: (C @ z).real + (self.D * u))(inner_states, x)

        return y


class LRUBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    lru: LRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, drop_rate, *, key):
        lrukey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        self.lru = LRULayer(state_dim, hidden_dim, r_min, r_max, theta_max, key=lrukey)
        self.glu = GLU(hidden_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.lru(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


class LRU(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: list[LRUBlock]
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification,
        tanh_output,
        output_step,
        r_min=0.9,
        r_max=1.0,
        theta_max=2 * jnp.pi,
        drop_rate=0.1,
        *,
        key,
    ):
        linear_encoder_key, *block_keys, linear_decoder_key = jr.split(
            key, num_blocks + 2
        )
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=linear_encoder_key)
        self.blocks = [
            LRUBlock(state_dim, hidden_dim, r_min, r_max, theta_max, drop_rate, key=key)
            for key in block_keys
        ]
        self.linear_decoder = eqx.nn.Linear(hidden_dim, output_dim, key=linear_decoder_key)
        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def __call__(self, x, state, key):
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)

        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = self.linear_decoder(x)
            x = jax.nn.softmax(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_decoder)(x)
            if self.tanh_output:
                x = jax.nn.tanh(x)

        return x, state
    