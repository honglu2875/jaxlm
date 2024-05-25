# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from .types import Array, DType, Shape


class KVCache(struct.PyTreeNode):
    """Simple pytree object for recording kv cache."""
    k: Array = struct.field(pytree_node=True)
    v: Array = struct.field(pytree_node=True)
    mask: Array = struct.field(pytree_node=True)
    dtype: DType = jnp.float32
    # kv cache is sometimes padded. end_pos indicate its ending position.
    end_pos: Array | int = -1
    # the number of axis that corresponds to the sequence direction.
    seq_axis: int = 1
    # kv cache may also have padding to the left, and one can apply a mask.

    def __post_init__(self):
        if isinstance(self.end_pos, jnp.ndarray) and self.end_pos.ndim != 1:
            raise ValueError(f"end_pos must be 1-dimensional. Got {self.end_pos.shape}.")
        if not (0 <= self.seq_axis < self.k.ndim):
            raise ValueError(f"seq_axis must be between 0 and {self.k.ndim}. Got {self.seq_axis}.")
        

    def _get_array(self, *args, start: int = 0, seq_axis: int = 1):
        if self.end_pos == -1 and start == 0:
            return args

        return tuple(map(
            lambda x: x.take(indices=jnp.arange(start, self.end_pos + 1), axis=seq_axis), 
            args, 
        ))

    def get_kv(self, start: int = 0):
        return self._get_array(self.k, self.v, start=start, seq_axis=self.seq_axis)
        

    def get_kv_mask(self, start: int = 0):
        return self._get_array(self.mask, start=start, seq_axis=self.seq_axis)

    def _check_shape(self, arr: Array, shape: Shape, name: str):
        if not len(shape) == len(arr.shape):
            raise ValueError(f"Unexpected shape {name=}{arr.shape}. Must have {len(shape)} axis.")
        if not all(s >= t for s, t in zip(shape, arr.shape)):
            raise ValueError(f"The given shape must be larger than {name} shapes. Got {shape} and {arr.shape}.")

    @classmethod
    def init(cls, *shape, k: Array | None = None, v: Array | None = None, mask: Array | None = None, dtype: DType = jnp.float32, seq_axis: int = 1):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        if k is not None and v is not None:
            self._check_shape(k, shape)
            self._check_shape(v, shape)
            
            k, v = jnp.pad(k, tuple((0, s - t) for s, t in zip(shape, k.shape)), constant_values=0), \
                   jnp.pad(v, tuple((0, s - t) for s, t in zip(shape, v.shape)), constant_values=0)
            end_pos = k.shape[seq_axis] - 1
        elif k is None and v is None:
            k, v = jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)
            end_pos = 0
        else:
            raise ValueError(f"One of (k, v) is None and the other is not. Got {k=} and {v=}")

        if mask is not None:
            self._check_shape(mask, shape)
            mask = jnp.pad(mask, tuple((0, s - t) for s, t in zip(shape, mask.shape)), constant_values=True, dtype=jnp.bool)
        else:
            mask = jnp.ones(shape, dtype=jnp.bool)

        return cls(k=k, v=v, dtype=dtype, end_pos=end_pos, seq_axis=seq_axis, mask=mask)

    def update(self, k: Array, v: Array, pos: Array | None = None):
        """Inplace update of k, v cache (at the mercy of JIT compiler).
        (Note: please jit-compile in order to have a chance of performing inplace update.)
        Arguments:
            k: the current k vectors (shape 1 at the sequence axis)
            v: the current v vectors (shape 1 at the sequence axis)
            pos: a 1-dim array specifying the index on sequence axis to update.
        """
        if pos is None:
            # If not provided with the update index, increment the recorded position.
            next_pos = self.end_pos + 1
            pos = self.end_pos
            if isinstance(pos, int):
                # Preserve the dimension
                index = tuple([slice(None)] * self.seq_axis + [jnp.array([pos])])
            else:
                index = tuple([slice(None)] * self.seq_axis + [pos])
        else:
            # If provided with the update index, use it.
            next_pos = pos + 1
            index = tuple([slice(None)] * self.seq_axis + [pos])



        return self.replace(k=self.k.at[index].set(k), v=self.v.at[index].set(v), end_pos=next_pos)

