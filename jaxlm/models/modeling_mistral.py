# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# This code is based on Hugging Face Mistral model code whose authors are
# denoted below. But it has been largely modified for JAX, Flax, and t5x.
# Original copyright message below:
#
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import functools
import warnings
from functools import partial
from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.partitioning import param_with_axes, with_sharding_constraint
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from t5x import partitioning as t5x_partitioning

from .._generate import generate
from ..nn.attention import Attention
from ..nn.linear import DenseGeneral
from ..nn.norms import RMSNorm
from ..nn.embedding import Embed
from ..outputs import BaseModelOutputWithCache, CausalLMOutputWithCache
from ..nn.position import RotaryEmbedding, apply_rotary_pos_emb
from ..types import Array
from ..utils import check_shape, get_default_pos_ids
from ..cache import KVCache


"""
# Copied from transformers.models.llama.modeling_llama._get_unpad_data
@jax.jit
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=jnp.int32)
    indices = jnp.nonzero(padding_mask.flatten(), as_tuple=False)[0].flatten()
    max_seqlen_in_batch = seqlens_in_batch.max()
    cu_seqlens = jnp.pad(jnp.cumsum(seqlens_in_batch, dim=0, dtype=jnp.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
"""


@partial(
    jax.jit,
    static_argnames=(
        "input_ids_shape",
        "dtype",
        #"past_key_values_length",
        "src_len",
        "sliding_window",
    ),
)
def _make_sliding_window_causal_mask(
    input_ids_shape: tuple,
    dtype: jnp.dtype,
    #past_key_values_length: int = 0,
    src_len: int,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    """
    if tgt_len == 1 and past_key_values_length > 0:
        # we are likely at inferencing stage and the causal mask can be fast-tracked
        pad_len = src_len - sliding_window - 1
        return jnp.log(
            jnp.triu(jnp.ones((1, src_len)), k=pad_len)
        )
    """

    tensor = jnp.ones(
        (tgt_len, tgt_len),
    )
    mask = jnp.tril(tensor, k=0)
    # make the mask banded to account for sliding window
    mask = jnp.triu(mask, k=-sliding_window)
    mask = jnp.log(mask).astype(dtype)

    #if src_len - tgt_len > 0:
    mask = jnp.concatenate(
        [jnp.zeros((tgt_len, src_len - tgt_len), dtype=dtype), mask], axis=-1
    )
    return jnp.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, src_len)
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
@partial(jax.jit, static_argnums=(1,), static_argnames=("tgt_len",))
def _expand_mask(mask: jnp.ndarray, dtype: jnp.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = jnp.broadcast_to(
        mask[:, None, None, :], (bsz, 1, tgt_len, src_len)
    ).astype(dtype)

    return jnp.where(expanded_mask == 0, jnp.finfo(dtype).min, 0.0).astype(dtype)


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.broadcast_to(
        hidden_states[:, :, :, None, :],
        (batch, slen, num_key_value_heads, n_rep, head_dim),
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class MistralMLP(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform
    act_fn: Any = jax.nn.silu

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        # input dim supposed to be self.hidden_size
        self.gate_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="gate_proj",
        )
        self.up_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="up_proj",
        )
        # input dim supposed to be self.intermediate_size
        self.down_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("intermediate", "embed"),
            name="down_proj",
        )

    def __call__(self, x, training=False):
        assert (
            x.shape[-1] == self.hidden_size
        ), f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        gate = self.act_fn(self.gate_proj(x))
        proj = self.up_proj(x)
        x = self.down_proj(gate * proj)
        return x


class MistralAttention(Attention):
    config: Any

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for attention.")

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        if self.fused_qkv:
            if self.num_heads != self.num_key_value_heads:
                raise ValueError(
                        f"If fusing qkv, num of heads must be the same as num of kv heads. "
                        f"Got {self.num_heads=} and {self.num_key_value_heads=}"
                )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_length=self.max_position_embeddings,
            base=self.rope_theta,
        )

        super().setup()

    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        kv_cache: Optional[Array] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        training: bool = False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[tuple]]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Input to Attention layer has different dimension than the hidden dimension. Got {hidden_states.shape[-1]}")

        bsz, q_len, _ = hidden_states.shape  # bsz, q_len, hidden_size

        # Obtain q, k, v from the current hidden state and shard q only (k, v will be handled later)
        query_states, key_states, value_states = self.qkv_proj(
            hidden_states
        )  # bsz, seq, n_head, head_dim

        past_kv_length = (
            kv_cache.end_pos if kv_cache is not None else 0
        )
        kv_seq_len = key_states.shape[-3] + past_kv_length
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids[:, past_kv_length:]
        )
        # attach kv-cache to k and v if exists, and shard k, v accordingly
        if kv_cache is not None:
            kv_cache = kv_cache.update(key_states, value_states)
            key_states, value_states = kv_cache.get_kv()

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jnp.einsum(
            "bshn,bthn->bhst", query_states, key_states
        ) / jnp.sqrt(self.head_dim)
        check_shape(attn_weights, bsz, self.num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            check_shape(attention_mask, bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            hidden_states.dtype
        )

        attn_output = jnp.einsum("bhst,bthn->bshn", attn_weights, value_states)
        check_shape(attn_output, bsz, q_len, self.num_heads, self.head_dim)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, kv_cache


class MistralDecoderLayer(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = MistralAttention(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.mlp = MistralMLP(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.input_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Array, Optional[Array], Optional[KVCache]]:
        """
        Args:
            hidden_states: input tensor
            attention_mask: mask for attention layer
            position_ids: position ids for positional embeddings
            kv_cache: kv cache
            output_attentions: whether to output attention weights
            use_cache: whether to use cached key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights if output_attentions else None, kv_cache


class MistralModel(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=self.vocab_size,
            features=self.config.hidden_size,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            one_hot=True,
            name="embed_tokens",
        )
        self.layers = [
            MistralDecoderLayer(
                self.config, dtype=self.dtype, kernel_init=self.kernel_init
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        #past_key_values_length,
        sliding_window,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = _make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            #past_key_values_length=past_key_values_length,
            src_len=attention_mask.shape[1],
            sliding_window=sliding_window,
        )

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

        return expanded_attn_mask + combined_attention_mask

    @staticmethod
    def _prepare_inference_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        sliding_window,
    ):
        """Create causal mask during inference.
        It will assume the query len being 1 and jit compiled to return fixed-length masks.
        Has to use a different kernel as _prepare_decoder_attention_mask is variable shape.
        """
        src_len = attention_mask.shape[1]
        combined_mask = attention_mask & (jnp.arange(src_len) >= src_len - sliding_window)
        return _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=1)

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithCache:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size, seq_length = input_ids.shape

        #past_key_values_length = (
        #    0 if kv_caches is None else kv_caches[0].end_pos
        #)

        if attention_mask is None:
            if kv_caches is not None:
                attention_mask = kv_caches[0].get_kv_mask(advance_right=input_ids.shape[1])
            else:
                attention_mask = jnp.ones(
                    (batch_size, seq_length), dtype=bool
                )

        if position_ids is None:
            if kv_caches is not None:
                position_ids = kv_caches[0].get_pos_ids(advance_right=input_ids.shape[1])
            else:
                position_ids = get_default_pos_ids((batch_size, seq_length), mask=attention_mask)

        inputs_embeds = self.embed_tokens(input_ids).astype(self.dtype)
        if seq_length == 1:
            attention_mask = self._prepare_inference_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                sliding_window=self.config.sliding_window,
            )
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                #past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = with_sharding_constraint(
            inputs_embeds, ("batch", "length", "embed")
        )

        all_hidden_states = []
        all_self_attns = []
        next_kv_caches = []

        for idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            kv_cache = None if kv_caches is None else kv_caches[idx]

            hidden_states, self_attn_weights, kv_cache  = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            all_self_attns.append(self_attn_weights)
            next_kv_caches.append(kv_cache)


        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return BaseModelOutputWithCache(
            last_hidden_state=hidden_states,
            kv_caches=tuple(next_kv_caches) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_self_attns) if output_attentions else None,
        )


class MistralForCausalLM(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform
    sharded: Optional[bool] = len(jax.devices()) > 1 and len(jax.devices()) % 2 == 0

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec | None, mesh: Mesh | None) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    @staticmethod
    def _parse_mesh_layout(device_mesh_layout):
        assert isinstance(device_mesh_layout, (list, tuple)), (
            f"device_mesh_layout must be a list or tuple. "
            f"Got {type(device_mesh_layout)}"
        )
        assert len(device_mesh_layout) == 2, (
            f"The length of device_mesh_layout must be 2. "
            f"Got {len(device_mesh_layout)}"
        )
        mesh_layout = []
        for i in range(2):
            if device_mesh_layout[i] is None:
                assert (
                    device_mesh_layout[1 - i] is not None
                ), f"Invalid device_mesh_layout. Got {device_mesh_layout}."
                mesh_layout.append(len(jax.devices()) // device_mesh_layout[1 - i])
            else:
                mesh_layout.append(device_mesh_layout[i])

        return tuple(mesh_layout)

    def _shard_params(self, x, y):
        if x.ndim != len(y.spec):
            assert (
                x.ndim == 2 and len(y.spec) == 3
            ), f"The shape of x ({x.shape}) and the sharding spec ({y.spec}) does not match"
            warnings.warn(
                f"The parameter has 2 axis ({x.shape}) while the sharding spec ({y.spec}) has 3 axis. "
                "Attempting to reshape into [:, :, head_dim], but please confirm that this is the intended behavior."
            )
            return jax.device_put(
                x.reshape(
                    (
                        x.shape[0],
                        -1,
                        self.config.hidden_size // self.config.num_attention_heads,
                    )
                ),
                y,
            )
        return (jax.device_put(x, y),)

    def init_cache(self, batch_size: int, max_len: int, mask: Optional[Array] = None) -> list[KVCache]:
        num_head = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers
        return [KVCache.init((batch_size, max_len, num_head, head_dim), mask=mask, left_buffer=max_len) for _ in range(num_layers)]

    def get_params(self, device_mesh_layout=(1, None), weights=None):
        """
        Get the properly sharded parameters.
        Args:
            device_mesh_layout: the device mesh layout. For example:
                (1, None) means data=1, model=len(jax.devices())
                (2, None) means data=2, model=len(jax.devices()) // 2
                (None, 2) means data=len(jax.devices()) // 2, model=2
            weights: whether a tree of weights are already given (but may not be sharded)
        Returns:
            a tree of properly sharded parameters
        """
        key = jax.random.PRNGKey(0)

        mesh_layout = self._parse_mesh_layout(device_mesh_layout)

        dummy_input = jnp.array(
            [[1 for _ in range(mesh_layout[1])] for _ in range(mesh_layout[0])]
        )

        abstract_variables = jax.eval_shape(self.init, key, dummy_input)
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(mesh_layout),
                axis_names=("data", "model"),
            )

            rules = t5x_partitioning.standard_logical_axis_rules(
                activation_partitioning_dims=1,
                parameter_partitioning_dims=1,
                additional_rules=(
                    ("kv_length", None),
                    ("intermediate", "model"),
                ),
            )
            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(
                logical_state_spec, mesh, rules
            )

            x_sharding = self.mesh_sharding(
                PartitionSpec("data", None), mesh
            )  # dimensions: (batch, length)

            if weights is not None:
                assert isinstance(
                    weights, dict
                ), f"weights must be a dict, got {type(weights)}"
                assert (
                    "params" in weights
                ), f"The key params not found in 'weights'. Got {weights.keys()}"

                if self.sharded:
                    params = {
                        "params": jax.tree_util.tree_map(
                            self._shard_params,
                            weights["params"],
                            logical_state_sharding["params"],
                        )
                    }
                else:
                    params = weights
            else:
                params = jax.jit(
                    self.init,
                    in_shardings=(
                        self.mesh_sharding(None, mesh),
                        x_sharding,
                    ),  # PRNG key and x
                    out_shardings=logical_state_sharding,
                )(key, dummy_input)
        else:
            params = self.init(key, dummy_input)

        return params

    def prepare_input(self, inputs, device_mesh_layout=(1, None), dtype=None):
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(
                    self._parse_mesh_layout(device_mesh_layout)
                ),
                axis_names=("data", "model"),
            )
            inputs = jax.device_put(
                inputs, self.mesh_sharding(PartitionSpec("data", None), mesh)
            )
        if dtype is not None:
            inputs = jax.tree_util.tree_map(lambda x: x.astype(dtype), inputs)
        return inputs

    def setup(self):
        self.model = MistralModel(
            self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[List[KVCache]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCache:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithCache(
            logits=logits,
            kv_caches=outputs.kv_caches,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def wrapped_apply_fn(
        self,
        params,
        tok,
        kv_caches=None,
        use_cache=True,
    ) -> tuple[CausalLMOutputWithCache, dict]:
        tok = jnp.array(tok)
        position_ids, attention_mask = None, None

        out, _ = self.apply(
            params,
            tok,
            position_ids=position_ids,
            attention_mask=attention_mask,
            mutable=("cache",),
            #output_hidden_states=False, # maybe allow for toggling of hidden states in the future
            #output_attentions=False, # maybe allow for toggling of attn wts in the future
            kv_caches=kv_caches,
            use_cache=use_cache,
        )  # return a tuple (CausalLMOutputWithCache, dict) where dict is the mutable cache

        return out.logits, out.kv_caches

    def generate(
        self,
        params,
        prompt_tokens: Array,
        attention_mask: Optional[Array] = None,
        do_sample: bool = True,
        seed: int = 0,
        max_tokens: int = 10,
        top_k: int = 0,
        top_p: float = 0.0,
        temp: float = 1.0,
        no_jit: bool = False,
    ):
        if no_jit:
            apply = self.wrapped_apply_fn
        else:
            apply = jax.jit(self.wrapped_apply_fn, static_argnames=("use_cache",))

        kv_caches = self.init_cache(
                batch_size=prompt_tokens.shape[0], 
                max_len=prompt_tokens.shape[1] + max_tokens,
                mask=attention_mask,
        )

        return generate(
            params,
            apply,
            prompt_tokens,
            kv_caches,
            do_sample=do_sample,
            seed=seed,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
