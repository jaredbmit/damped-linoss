import abc
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from damped_linoss.models.common import GLU


class _AbstractRNNCell(eqx.Module):
    """Abstract RNN Cell class."""
    cell: eqx.Module

    @abc.abstractmethod
    def __init__(self, data_dim, hidden_dim, depth, width, *, key):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, state, input):
        raise NotImplementedError


class GRUCell(_AbstractRNNCell):
    cell: eqx.nn.GRUCell

    def __init__(self, input_dim, hidden_dim, *, key):
        self.cell = eqx.nn.GRUCell(input_dim, hidden_dim, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))

        scan_fn = lambda state, input: (
            self.cell(input, state),
            self.cell(input, state),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states


class LSTMCell(_AbstractRNNCell):
    cell: eqx.nn.LSTMCell

    def __init__(self, input_dim, hidden_dim, *, key):
        self.cell = eqx.nn.LSTMCell(input_dim, hidden_dim, key=key)

    def __call__(self, x):
        hidden = jnp.zeros((self.cell.hidden_size,))
        hidden = (hidden,) * 2

        scan_fn = lambda state, input: (
            self.cell(input, state),
            self.cell(input, state),
        )
        _, all_states = jax.lax.scan(scan_fn, hidden, x)

        return all_states[0]  # LSTM specific


class BasicRNN(eqx.Module):
    linear_encoder: eqx.nn.Linear
    cell: _AbstractRNNCell
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = False
    nondeterministic: bool = False

    def __init__(
        self,
        cell_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        classification,
        tanh_output,
        output_step,
        *,
        key
    ):
        input_key, cell_key, output_key = jr.split(key, 3)
        cell_map = {
            "LSTM": LSTMCell,
            "GRU": GRUCell,
        }

        self.linear_encoder = eqx.nn.Linear(
            input_dim, hidden_dim, key=input_key
        )
        self.cell = cell_map[cell_name](
            hidden_dim, state_dim, key=cell_key
        )
        self.linear_decoder = eqx.nn.Linear(
            state_dim, output_dim, key=output_key
        )

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def __call__(self, x):
        x = jax.vmap(self.linear_encoder)(x)

        x = self.cell(x)

        if self.classification:
            x = jnp.mean(x, axis=0)
            x = self.linear_decoder(x)
            x = jax.nn.softmax(x, axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.vmap(self.linear_decoder)(x)
            if self.tanh_output:
                x = jax.nn.tanh(x)

        return x


class RNNBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    cell: _AbstractRNNCell
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self, 
        cell_type, 
        state_dim, 
        hidden_dim, 
        drop_rate, 
        *, 
        key
    ):
        cellkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        self.cell = cell_type(hidden_dim, state_dim, key=cellkey)
        self.glu = GLU(state_dim, hidden_dim, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.cell(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x

        return x, state


class StackedRNN(eqx.Module):
    """
    Multi-layer deep RNN with GLU activation, skip connections, and dropout.
    """
    linear_encoder: eqx.nn.Linear
    blocks: list[RNNBlock]
    linear_decoder: eqx.nn.Linear
    classification: bool
    tanh_output: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        cell_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification,
        tanh_output,
        output_step,
        drop_rate=0.1,
        *,
        key
    ):
        cell_map = {
            "LSTM": LSTMCell,
            "GRU": GRUCell,
        }
        cell_type = cell_map[cell_name]

        linear_encoder_key, *block_keys, linear_decoder_key = jr.split(key, num_blocks + 2)
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=linear_encoder_key)
        self.blocks = [
            RNNBlock(cell_type, state_dim, hidden_dim, drop_rate, key=key)
            for key in block_keys
        ]
        self.linear_decoder = eqx.nn.Linear(
            hidden_dim, output_dim, key=linear_decoder_key
        )
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
