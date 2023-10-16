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
from functools import partial
from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from t5x import partitioning as t5x_partitioning
from t5x.examples.t5 import layers

from ._generate import generate
from .activations import ACT2FN

"""
Notes:
It uses t5x.examples.t5.layers so that it is compatible with the t5 library. But t5x defines logical named axis
and operates sharding in a different fashion than the flax official way of using `nn.with_logical_partitioning`...
Although I hate doing this I am mixing both ways. 
Putting this note out there to say that it's not my fault for this code. For any serious use, this needs a lot 
of refactoring. I have already cleaned up some mess from the insane Hugging Face `modeling_mistral.py` and
hopefully things are not too difficult from here.
"""


@flax.struct.dataclass
class BaseModelOutputWithPast:
    last_hidden_state: jnp.ndarray
    past_key_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@flax.struct.dataclass
class CausalLMOutputWithPast:
    logits: jnp.ndarray
    past_key_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


def _check_shape(tensor, *shape):
    chex.assert_shape(tensor, shape)


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


@partial(
    jax.jit,
    static_argnames=(
        "input_ids_shape",
        "dtype",
        "past_key_values_length",
        "sliding_window",
    ),
)
def _make_sliding_window_causal_mask(
    input_ids_shape: tuple,
    dtype: jnp.dtype,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = jnp.full(
        (tgt_len, tgt_len),
        fill_value=1,
    )
    mask = jnp.tril(tensor, k=0)
    # make the mask banded to account for sliding window
    mask = jnp.triu(mask, k=-sliding_window)
    mask = jnp.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate(
            [jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1
        )
    return jnp.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length)
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


param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


class MistralRMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = param_with_axes(
            "weight",
            nn.with_logical_partitioning(
                lambda _, shape, dtype: jnp.ones(shape, dtype=dtype), ("embed",)
            ),
            (self.hidden_size,),
            jnp.float32,
            axes=("embed",),
        )
        self.variance_epsilon = self.eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.square(hidden_states).mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class MistralRotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: int = 10000

    def setup(self):
        self.inv_freq = self.variable(
            "cache",
            "inv_freq",
            lambda: 1.0
            / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)),
        )

        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, dtype=jnp.float32)

    @nn.compact
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached, dtype=jnp.int32)

        freqs = jnp.einsum("i,j->ij", t, self.inv_freq.value)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        is_initialized = self.has_variable("cache", "cos_cached")
        self.cos_cached = self.variable(
            "cache", "cos_cached", lambda: jnp.cos(emb).astype(dtype)
        )
        self.sin_cached = self.variable(
            "cache", "sin_cached", lambda: jnp.sin(emb).astype(dtype)
        )
        if is_initialized:
            self.cos_cached.value = jnp.cos(emb).astype(dtype)
            self.sin_cached.value = jnp.sin(emb).astype(dtype)

    def __call__(self, x: jnp.ndarray, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=jnp.float32)

        return (
            self.cos_cached.value[:seq_len].astype(x.dtype),
            self.sin_cached.value[:seq_len].astype(x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
@jax.jit
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
@jax.jit
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
    cos = jnp.expand_dims(jnp.take(cos, position_ids, axis=0), axis=1)
    sin = jnp.expand_dims(jnp.take(sin, position_ids, axis=0), axis=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@partial(jax.jit, static_argnames=("n_rep",))
def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.broadcast_to(
        hidden_states[:, :, None, :, :],
        (batch, num_key_value_heads, n_rep, slen, head_dim),
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralMLP(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        # input dim supposed to be self.hidden_size
        self.gate_proj = layers.DenseGeneral(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "intermediate")
            ),
            kernel_axes=("intermediate", "embed"),
            name="gate_proj",
        )
        self.up_proj = layers.DenseGeneral(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("intermediate", "up_sample")
            ),
            kernel_axes=("intermediate", "up_sample"),
            name="up_proj",
        )
        # input dim supposed to be self.intermediate_size
        self.down_proj = layers.DenseGeneral(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("up_sample", "embed")
            ),
            kernel_axes=("up_sample", "embed"),
            name="down_proj",
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x, training=False):
        assert (
            x.shape[-1] == self.hidden_size
        ), f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralAttention(nn.Module):
    """
    Flax implementation of attention.
    """

    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

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
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # input dim supposed to be self.hidden_size
        self.q_proj = layers.DenseGeneral(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="q_proj",
        )
        self.k_proj = layers.DenseGeneral(
            self.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="k_proj",
        )
        self.v_proj = layers.DenseGeneral(
            self.num_key_value_heads * self.head_dim,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "joined_kv")
            ),
            kernel_axes=("embed", "joined_kv"),
            name="v_proj",
        )
        self.o_proj = layers.DenseGeneral(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("joined_kv", "embed")
            ),
            kernel_axes=("joined_kv", "embed"),
            name="o_proj",
        )

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    @jax.jit
    def _shape(self, tensor: jnp.ndarray, seq_len: int, bsz: int):
        return jnp.swapaxes(
            tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim), 1, 2
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        padding_mask=None,
        training=False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[tuple]]:
        assert (
            hidden_states.shape[-1] == self.hidden_size
        ), f"Input to Attention layer has different dimension than the hidden dimension. Got {hidden_states.shape[-1]}"

        bsz, q_len = hidden_states.shape[-3:-1]  # bsz, q_len, hidden_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = jnp.swapaxes(
            query_states.reshape(bsz, q_len, self.num_heads, self.head_dim), 1, 2
        )
        key_states = jnp.swapaxes(
            key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim),
            1,
            2,
        )
        value_states = jnp.swapaxes(
            value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim),
            1,
            2,
        )
        query_states = with_sharding_constraint(
            query_states, ("batch", "heads", "length", "kv")
        )
        key_states = with_sharding_constraint(
            key_states, ("batch", "heads", "kv_length", "kv")
        )
        value_states = with_sharding_constraint(
            value_states, ("batch", "heads", "kv_length", "kv")
        )

        kv_seq_len = key_states.shape[-2] + (
            past_key_value[0].shape[-2] if past_key_value is not None else 0
        )
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should be a tuple of (k, v)"
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=2)
            value_states = jnp.concatenate([past_value, value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = with_sharding_constraint(
            key_states, ("batch", "heads", "kv_length", "kv")
        )
        value_states = with_sharding_constraint(
            value_states, ("batch", "heads", "kv_length", "kv")
        )

        attn_weights = (query_states @ jnp.swapaxes(key_states, 2, 3)) / jnp.sqrt(
            self.head_dim
        )

        _check_shape(attn_weights, bsz, self.num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            _check_shape(attention_mask, bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            hidden_states.dtype
        )
        attn_output = attn_weights @ value_states

        _check_shape(attn_output, bsz, self.num_heads, q_len, self.head_dim)

        attn_output = jnp.swapaxes(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class MistralDecoderLayer(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = MistralAttention(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.mlp = MistralMLP(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.input_layernorm = MistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.post_attention_layernorm = MistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple:
        """
        Args:
            hidden_states: input tensor
            attention_mask: mask for attention layer
            position_ids: position ids for positional embeddings
            past_key_value: cached key and value projection states
            output_attentions: whether to output attention weights
            use_cache: whether to use cached key and value projection states
            padding_mask: mask for padding
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralModel(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = layers.Embed(
            num_embeddings=self.vocab_size,
            features=self.config.hidden_size,
            attend_dtype=self.dtype,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=1.0),
                (
                    "vocab",
                    "embed",
                ),
            ),
            one_hot=True,
            name="embed_tokens",
        )
        self.layers = [
            MistralDecoderLayer(
                self.config, dtype=self.dtype, kernel_init=self.kernel_init
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = MistralRMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = _make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

        return expanded_attn_mask + combined_attention_mask

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
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

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = jnp.expand_dims(
                jnp.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=jnp.int32,
                ),
                0,
            )

        padding_mask = None

        # embed positions
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length_with_past), dtype=bool)
        else:
            padding_mask = attention_mask

        inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

        hidden_states = with_sharding_constraint(
            inputs_embeds, ("batch", "length", "embed")
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform()

    sharded: Optional[bool] = len(jax.devices()) > 1 and len(jax.devices()) % 2 == 0
    device_mesh: Optional[np.ndarray] = (
        mesh_utils.create_device_mesh((2, len(jax.devices()) // 2)) if sharded else None
    )
    mesh: Optional[Mesh] = (
        Mesh(devices=device_mesh, axis_names=("data", "model")) if sharded else None
    )

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec | None, mesh: Mesh | None) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    def get_params(self, weights=None):
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.array([[1, 1], [1, 1]])
        abstract_variables = jax.eval_shape(self.init, key, dummy_input)
        if self.sharded:
            rules = t5x_partitioning.standard_logical_axis_rules(
                activation_partitioning_dims=1,
                parameter_partitioning_dims=1,
                additional_rules=(
                    ("kv_length", None),
                    ("intermediate", None),
                    ("up_sample", "model"),
                ),
            )
            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(
                logical_state_spec, self.mesh, rules
            )

            x_sharding = self.mesh_sharding(
                PartitionSpec("data", None), self.mesh
            )  # dimensions: (batch, length)

            params = jax.jit(
                self.init,
                in_shardings=(
                    self.mesh_sharding(None, self.mesh),
                    x_sharding,
                ),  # PRNG key and x
                out_shardings=logical_state_sharding,
            )(key, dummy_input)
        else:
            params = self.init(key, dummy_input)

        if weights is not None:
            assert isinstance(
                weights, dict
            ), f"weights must be a dict, got {type(weights)}"
            assert (
                "params" in weights
            ), f"The key params not found in 'weights'. Got {weights.keys()}"
            if self.sharded:
                params.update(
                    {
                        "params": jax.jit(
                            lambda: weights["params"],
                            in_shardings=None,
                            out_shardings=logical_state_sharding["params"],
                        )()
                    }
                )
            else:
                params.update(weights)

        return params

    def prepare_input(self, inputs):
        if self.sharded:
            inputs = jax.device_put(
                inputs, self.mesh_sharding(PartitionSpec("data", None), self.mesh)
            )
        return inputs

    def setup(self):
        self.model = MistralModel(
            self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.lm_head = layers.DenseGeneral(
            self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                self.kernel_init, ("embed", "vocab")
            ),
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        params,
        prompt_tokens: list | jnp.ndarray,
        do_sample: bool = True,
        seed: int = 0,
        max_length: int = 10,
        top_k: int = 0,
        top_p: float = 0.0,
        temp: float = 1.0,
        no_jit: bool = False,
    ):
        if no_jit:
            apply = functools.partial(
                self.apply, mutable=("cache",), output_hidden_states=False
            )
        else:
            apply = jax.jit(
                functools.partial(
                    self.apply, mutable=("cache",), output_hidden_states=False
                ),
                static_argnames=("use_cache",),
            )

        def apply_fn(
            params, tok, attention_mask=None, past_key_values=None, use_cache=True
        ):
            out = apply(
                params,
                jnp.array(tok),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )[
                0
            ]  # return a tuple (CausalLMOutputWithPast, dict) where dict is the mutable cache
            return out.logits, out.past_key_values

        return generate(
            params,
            apply_fn,
            prompt_tokens,
            do_sample=do_sample,
            seed=seed,
            max_len=max_length,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
