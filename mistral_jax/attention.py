#  Copyright 2024 Honglu Fan
#  This file is based on code by the authors denoted below and has been modified from its original version.
#
#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# This file has been modified from its original version
# Link: https://github.com/google/maxtext/blob/4f3a0d3cf8509d05ce040e35d88ea7bf57797945/MaxText/layers/attentions.py
#

import functools
import math
from typing import Optional, Sequence

from flax import linen as nn
import jax
from jax import lax
from jax import random
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops import attention as pallas_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
import jax.numpy as jnp

from .types import Array, Config, DType, Mesh, PRNGKey, AxisNames, BATCH, LENGTH, HEAD, D_KV


#DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class AttentionOp(nn.Module):
    mesh: Mesh
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    float32_qk_product: bool = False
    max_prefill_predict_length: int = -1
    float32_logits: bool = False
    axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
    dropout_rate: float = 0.0
    dtype: DType = jnp.float32

    def check_attention_inputs(self, query: Array, key: Array, value: Array) -> None:
        """Check attention inputs."""

        assert key.ndim == value.ndim, "k, v must have same rank."
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
        assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
        assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
        assert query.shape[-1] == key.shape[-1], "q, k depths must match."


class FlashAttentionOp(AttentionOp):

    def tpu_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None) -> Array:
        """TPU Flash Attention."""
        # Transpose to ('batch', 'heads', 'length', 'kv')
        query = jnp.transpose(query, axes=(0, 2, 1, 3))
        key = jnp.transpose(key, axes=(0, 2, 1, 3))
        value = jnp.transpose(value, axes=(0, 2, 1, 3))

        if decoder_segment_ids is not None:
            decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
        axis_names = nn.logical_to_mesh_axes(self.axis_names)
        segment_axis_names = nn.logical_to_mesh_axes((BATCH, "activation_length_no_heads"))

        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(
                axis_names,
                axis_names,
                axis_names,
                segment_axis_names,
            ),
            out_specs=axis_names,
            check_rep=False,
        )
        def wrap_flash_attention(query, key, value, decoder_segment_ids):
            if decoder_segment_ids is not None:
                assert (
                    query.shape[2] == decoder_segment_ids.q.shape[1]
                ), "Sharding along sequence dimension not allowed in tpu kernel attention"
            block_sizes = splash_attention_kernel.BlockSizes(
                block_q=min(512, query.shape[2]),
                block_kv_compute=min(512, key.shape[2]),
                block_kv=min(512, key.shape[2]),
                block_q_dkv=min(512, query.shape[2]),
                block_kv_dkv=min(512, key.shape[2]),
                block_kv_dkv_compute=min(512, query.shape[2]),
                block_q_dq=min(512, query.shape[2]),
                block_kv_dq=min(512, query.shape[2]),
            )

            masks = [splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2])) for i in range(query.shape[1])]
            multi_head_mask = splash_attention_mask.MultiHeadMask(masks=masks)
            splash_kernel = splash_attention_kernel.make_splash_mha(
                mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
            )

            return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

        #devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
        #assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        #    "Batch dimension should be shardable among the devices in data and fsdp" " axis"
        #)
        x = wrap_flash_attention(query, key, value, decoder_segment_ids)
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x

    def __call__(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None = None) -> Array:
        self.check_attention_inputs(query, key, value)
        output = self.tpu_flash_attention(query, key, value, decoder_segment_ids=decoder_segment_ids)
        return output
    


