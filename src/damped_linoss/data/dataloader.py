"""
This module implements `Dataloader` classes, which loop and yield batches
during runtime.

It supports four different subclasses, tailored for use
with different types of models available in this repository.

1. `StandardDataloader`: Time series data with fixed-length sequences.
    Used by RNNs, SSMs, and the like.

2. `BucketedDataloader`: Time series data with variable-length sequences.
    Contains extra functionality for batching and processing during training time.
    Used by RNNs, SSMs, and the like.

Additionally, data can be stored as a NumPy array to save GPU memory,
with each batch converted to a JAX NumPy array.

Public Class Methods:
- `loop(batch_size, *, key)`: Generates data batches indefinitely.
                              Randomly shuffles data for each batch.
- `loop_epoch(batch_size)`: Generates data batches for one epoch
                            (i.e., a full pass through the dataset).
"""
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax.random as jr


class BaseDataloader(ABC):
    def __init__(self, data, labels, in_memory):
        self.data = data
        self.labels = labels
        self.size = self._calculate_size()

        if in_memory:
            self.func = lambda x: x
        else:
            self.func = lambda x: jnp.asarray(x)

    def __iter__(self):
        RuntimeError("Use .loop(batch_size) instead of __iter__")

    @abstractmethod
    def _calculate_size(self):
        pass

    @abstractmethod
    def _loop(self, batch_size, *, key):
        pass

    @abstractmethod
    def _loop_epoch(self, batch_size):
        pass

    def loop(self, batch_size, *, key):
        if self.size == 0:
            raise ValueError("This dataloader is empty")

        if not (isinstance(batch_size, int) and (batch_size > 0)):
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")

        return self._loop(batch_size, key=key)

    def loop_epoch(self, batch_size):
        if self.size == 0:
            raise ValueError("This dataloader is empty")

        if not (isinstance(batch_size, int) and (batch_size > 0)):
            raise ValueError("Batch size must be a positive integer")

        if batch_size > self.size:
            raise ValueError("Batch size larger than dataset size")

        return self._loop_epoch(batch_size)


class StandardDataloader(BaseDataloader):
    """
    Batches and yields data that can be stored as a single block array.
    Input data should be a jnp.ndarray of shape (n_samples, n_timesteps, n_features).
    """
    def __init__(self, data, labels, in_memory, data_out_func):
        super().__init__(data, labels, in_memory)

        self.data_out_func = data_out_func

    def _calculate_size(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

    def _loop(self, batch_size, *, key):
        while True:
            batch_key, key = jr.split(key)
            if len(self.data) <= batch_size:
                yield (
                    self.data_out_func(self.func(self.data)),
                    self.func(self.labels),
                )
            else:
                idxs = jr.choice(key, self.size, shape=(batch_size,), replace=False)
                yield (
                    self.data_out_func(self.func(self.data[idxs])),
                    self.func(self.labels[idxs]),
                )

    def _loop_epoch(self, batch_size):
        if len(self.data) <= batch_size:
            yield (
                self.data_out_func(self.func(self.data)),
                self.func(self.labels),
            )
        else:
            start = 0
            end = batch_size
            while end < self.size:
                idxs = jnp.arange(start, end)
                yield (
                    self.data_out_func(self.func(self.data[idxs])),
                    self.func(self.labels[idxs]),
                )
                start = end
                end = start + batch_size

            # Remainder
            idxs = jnp.arange(start, self.size)
            yield (
                self.data_out_func(self.func(self.data[idxs])),
                self.func(self.labels[idxs]),
            )


class BucketedDataloader(BaseDataloader):
    """
    Pre-batches data into buckets by sequence length, yields randomly chosen batches.
    Input data should be a list of jnp.ndarray's of shape (n_timesteps, n_features).
    Where n_timesteps are not necessarily the same length.
    """
    def __init__(
        self,
        data,
        labels,
        in_memory,
        data_out_func,
        bucket_boundaries=None,
    ):
        super().__init__(data, labels, in_memory)
        if bucket_boundaries is None:
            # Default boundaries
            bucket_boundaries = [100, 200, 400, 800, 1600, 3200, 6400]

        self.buckets = self._create_buckets(bucket_boundaries)
        self.bucket_sizes = [len(b[0]) for b in self.buckets]

        self.data_out_func = data_out_func

    def _calculate_size(self):
        if self.data is None:
            return 0
        else:
            return len(self.data)

    def _create_buckets(self, boundaries):
        buckets_data = [[] for _ in range(len(boundaries) + 1)]
        buckets_labels = [[] for _ in range(len(boundaries) + 1)]
        buckets_indices = [[] for _ in range(len(boundaries) + 1)]

        # Sort sequences into buckets
        for i, (seq, label) in enumerate(zip(self.data, self.labels)):
            length = len(seq)
            for j, boundary in enumerate(boundaries):
                if length <= boundary:
                    buckets_data[j].append(seq)
                    buckets_labels[j].append(label)
                    buckets_indices[j].append(i)
                    break
            else:
                buckets_data[-1].append(seq)
                buckets_labels[-1].append(label)
                buckets_indices[-1].append(i)

        # Pad sequences in each bucket
        padded_buckets = []
        for bucket_data, bucket_labels, bucket_indices in zip(
            buckets_data, buckets_labels, buckets_indices
        ):
            if not bucket_data:  # Skip empty buckets
                continue

            max_length = max(len(seq) for seq in bucket_data)

            # Pad all sequences to same length
            padded_seqs = []
            for seq in bucket_data:
                if seq.ndim == 1:
                    num_dim = 1
                else:
                    num_dim = seq.shape[1]

                padded_seq = np.pad(
                    seq.reshape(-1, num_dim),
                    pad_width=((0, max_length - len(seq)), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                padded_seqs.append(padded_seq)

            padded_seqs = jnp.asarray(np.array(padded_seqs))
            bucket_labels = jnp.asarray(np.array(bucket_labels))

            padded_buckets.append((padded_seqs, bucket_labels, bucket_indices))

        return padded_buckets

    def _loop(self, batch_size, *, key):
        while True:
            batch_key, key = jr.split(key)

            bucket_probs = jnp.array(self.bucket_sizes) / sum(self.bucket_sizes)
            bucket_idx = jr.choice(batch_key, len(self.buckets), p=bucket_probs)
            bucket_data, bucket_labels, _ = self.buckets[bucket_idx]

            if len(bucket_data) <= batch_size:
                yield (
                    self.data_out_func(self.func(bucket_data)),
                    self.func(bucket_labels),
                )
            else:
                idxs = jr.choice(
                    batch_key, len(bucket_data), shape=(batch_size,), replace=False
                )
                yield (
                    self.data_out_func(self.func(bucket_data[idxs])),
                    self.func(bucket_labels[idxs]),
                )

    def _loop_epoch(self, batch_size):
        for bucket_data, bucket_labels, _ in self.buckets:
            bucket_size = len(bucket_data)
            if len(bucket_data) <= batch_size:
                yield (
                    self.data_out_func(self.func(bucket_data)),
                    self.func(bucket_labels),
                )
            else:
                start = 0
                end = batch_size
                while end < bucket_size:
                    idxs = jnp.arange(start, end)
                    yield (
                        self.data_out_func(self.func(bucket_data[idxs])),
                        self.func(bucket_labels[idxs]),
                    )
                    start = end
                    end = start + batch_size

                # Remainder
                idxs = jnp.arange(start, bucket_size)
                yield (
                    self.data_out_func(self.func(bucket_data[idxs])),
                    self.func(bucket_labels[idxs]),
                )
