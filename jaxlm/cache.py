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

from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from .types import Array, DType, Shape
from .utils import get_default_pos_ids


class KVCache(struct.PyTreeNode):
    """Simple pytree object for recording kv cache."""
    k: Array = struct.field(pytree_node=True)
    v: Array = struct.field(pytree_node=True)
    mask: Array = struct.field(pytree_node=True)
    pos_ids: Array = struct.field(pytree_node=True)
    dtype: DType = struct.field(pytree_node=False, default=jnp.float32)
    # kv cache is sometimes padded. end_pos indicate its ending position.
    end_pos: int = struct.field(pytree_node=True, default=-1)
    offset: int = struct.field(pytree_node=False, default=0)

    def _get_array(self, *args, full_len: Optional[int] = None, advance_right: int = 0):
        full_len = full_len or self.offset
        start_idx = self.end_pos + self.offset - full_len + advance_right
        return tuple(map(
            #lambda x: x[:, self.end_pos + start_idx : start_idx + full_len + advance_right], 
            lambda x: jax.lax.dynamic_slice(x, (0, start_idx) + (0,) * (len(x.shape) - 2), (x.shape[0], full_len) + x.shape[2:]),
            args, 
        ))

    def get_kv(self, full_len: Optional[int] = None):
        return self._get_array(self.k, self.v, full_len=full_len)
        
    def get_kv_mask(self, full_len: Optional[int] = None, advance_right: int = 0):
        return self._get_array(self.mask, full_len=full_len, advance_right=advance_right)[0]

    def get_pos_ids(self, full_len: Optional[int] = None, advance_right: int = 0):
        return self._get_array(self.pos_ids, full_len=full_len, advance_right=advance_right)[0]

    @classmethod
    def init(cls, shape: tuple, k: Optional[Array] = None, v: Optional[Array] = None, left_buffer: Optional[int] = None, mask: Optional[Array] = None, pos_ids: Optional[Array] = None, dtype: DType = jnp.float32):
        extra_len = left_buffer if left_buffer is not None else shape[1]
        full_shape = (shape[0], extra_len + shape[1]) + shape[2:]
        if k is None and v is None:
            k, v = jnp.zeros(full_shape, dtype=dtype), jnp.zeros(full_shape, dtype=dtype)
            end_pos = 0
        else:
            k, v = jnp.pad(k, ((0, 0), (extra_len, shape[1] - k.shape[1])) + ((0, 0)) * (len(shape) - 2), constant_values=0), \
                   jnp.pad(k, ((0, 0), (extra_len, shape[1] - k.shape[1])) + ((0, 0)) * (len(shape) - 2), constant_values=0)
            end_pos = k.shape[1]

        if mask is not None:
            head = jnp.zeros((shape[0], extra_len), dtype=jnp.bool)
            tail = jnp.ones((shape[0], shape[1] - mask.shape[1]), dtype=jnp.bool)
            mask = jnp.concatenate((head, mask, tail), axis=1)
        else:
            mask = jnp.ones(full_shape[:2], dtype=jnp.bool)

        if pos_ids is None:
            pos_ids = get_default_pos_ids(mask.shape, mask=mask)

        return cls(k=k, v=v, dtype=dtype, end_pos=end_pos, mask=mask, pos_ids=pos_ids, offset=extra_len)

    def update(self, k: Array, v: Array):
        """Inplace update of k, v cache (at the mercy of JIT compiler).
        (Note: please jit-compile in order to have a chance of performing inplace update.)
        Arguments:
            k: the current k vectors (shape 1 at the sequence axis)
            v: the current v vectors (shape 1 at the sequence axis)
        """
        next_pos = self.end_pos + k.shape[1]
        index = (slice(None), jnp.arange(k.shape[1]) + self.end_pos + self.offset)

        return self.replace(k=self.k.at[index].set(k), v=self.v.at[index].set(v), end_pos=next_pos)

    """
    def extend_to(self, extend_len: int) -> "KVCache":
        bs, seq_len = self.k.shape[:2]
        extra_len = extend_len - seq_len
        pad = jnp.zeros((bs, extra_len), dtype=self.k.dtype)
        k, v = map(lambda x: jnp.concatenate((x, pad), axis=1), (self.k, self.v))
        mask = jnp.concatenate((self.mask, jnp.ones((bs, extra_len), jnp.bool)), axis=1)
        extra_pos = self.pos_ids[:, -1:] + jnp.arange(extra_len)[None] + 1
        pos_ids = jnp.concatenate((self.pos_ids, extra_pos), axis=1)
        return self.replace(k=k, v=v, mask=mask, pos_ids=pos_ids)
    """
