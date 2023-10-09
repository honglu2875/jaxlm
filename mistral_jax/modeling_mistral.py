from functools import partial, lru_cache
from typing import Optional, Tuple, Union, List, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jax import lax
from flax.linen import partitioning as nn_partitioning
import chex
from .activations import ACT2FN


# TODO:
#  1. attn_mask,
#  2. different attn_impl


@flax.struct.dataclass
class BaseModelOutputWithPast:
    last_hidden_state: jnp.ndarray
    past_key_values: Optional[Tuple[jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None


def _check_shape(tensor, *shape):
    if tensor.shape != shape:
        raise ValueError(
            f"`{tensor.__name__}` should be of size {shape}, but is"
            f" {tensor.shape}"
        )


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


@lru_cache(maxsize=32)
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
            [jnp.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1
        )
    return jnp.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))


# Copied from transformers.models.bart.modeling_bart._expand_mask
@partial(jax.jit, static_argnums=(1,), static_argnames=('tgt_len',))
def _expand_mask(mask: jnp.ndarray, dtype: jnp.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = jnp.broadcast_to(mask[:, None, None, :], (bsz, 1, tgt_len, src_len)).astype(dtype)

    return jnp.where(expanded_mask == 0, -jnp.inf, 1.0).astype(dtype)


param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


class MistralRMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        self.weight = self.param('weight', lambda rng, shape: jnp.ones(shape), (self.hidden_size,))
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
        self.inv_freq = self.variable("cache", "inv_freq",
                                      lambda: 1.0 / (self.base **
                                                     (jnp.arange(0, self.dim, 2,
                                                                 dtype=jnp.float32) / self.dim)
                                                     ))

        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, dtype=jnp.float32
        )

    @nn.compact
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached, dtype=jnp.int32)

        freqs = jnp.einsum("i,j->ij", t, self.inv_freq.value)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        is_initialized = self.has_variable("cache", "cos_cached")
        self.cos_cached = self.variable('cache', 'cos_cached', lambda: jnp.cos(emb).astype(dtype))
        self.sin_cached = self.variable('cache', 'sin_cached', lambda: jnp.sin(emb).astype(dtype))
        if is_initialized:
            self.cos_cached.value = jnp.cos(emb).astype(dtype)
            self.sin_cached.value = jnp.sin(emb).astype(dtype)

    def __call__(self, x: jnp.ndarray, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached.value[:seq_len].astype(x.dtype),
            self.sin_cached.value[:seq_len].astype(x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
@jax.jit
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
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


@partial(jax.jit, static_argnames=('n_rep',))
def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.broadcast_to(hidden_states[:, :, None, :, :],
                                     (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralMLP(nn.Module):
    config: Any = None

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        # input dim supposed to be self.hidden_size
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False)
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False)
        # input dim supposed to be self.intermediate_size
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, x, training=False):
        assert x.shape[-1] == self.hidden_size, \
            f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MistralAttention(nn.Module):
    """
    Flax implementation of attention.
    """
    config: Any = None

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
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False)
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False)
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=False)
        # input dim supposed to be self.num_heads * self.head_dim
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    @jax.jit
    def _shape(self, tensor: jnp.ndarray, seq_len: int, bsz: int):
        return jnp.swapaxes(
            tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim),
            1, 2
        )

    def __call__(self,
                 hidden_states,
                 attention_mask=None,
                 position_ids=None,
                 past_key_value=None,
                 output_attentions=False,
                 use_cache=False,
                 padding_mask=None,
                 training=False) \
            -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[tuple]]:
        assert hidden_states.shape[-1] == self.hidden_size, \
            f"Input to Attention layer has different dimension than the hidden dimension. Got {hidden_states.shape[-1]}"

        bsz, q_len = hidden_states.shape[-3:-1]  # bsz, q_len, hidden_size

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = jnp.swapaxes(
            query_states.reshape(bsz, q_len, self.num_heads, self.head_dim),
            1, 2
        )
        key_states = jnp.swapaxes(
            key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim),
            1, 2
        )
        value_states = jnp.swapaxes(
            value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim),
            1, 2
        )

        kv_seq_len = key_states.shape[-2] + (past_key_value[0].shape[-2] if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            assert len(past_key_value) == 2, 'past_key_value should be a tuple of (k, v)'
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=2)
            value_states = jnp.concatenate([past_value, value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (query_states @ jnp.swapaxes(key_states, 2, 3)) / jnp.sqrt(self.head_dim)

        _check_shape(attn_weights, bsz, self.num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            _check_shape(attention_mask, bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
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

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = MistralAttention(config=self.config)
        self.mlp = MistralMLP(self.config)
        self.input_layernorm = MistralRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

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

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size, self.padding_idx)
        self.layers = [MistralDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        self.norm = MistralRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        # self.embed_tokens = self.variable('cache', 'embed_tokens', lambda rng, shape: jnp.zeros(shape))

        self.gradient_checkpointing = False

    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds, past_key_values_length, sliding_window
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_sliding_window_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
                sliding_window=sliding_window,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = jnp.expand_dims(jnp.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=jnp.int32
            ), 0)

        padding_mask = None

        # embed positions
        if attention_mask is None:
            attention_mask = jnp.ones(
                (batch_size, seq_length_with_past), dtype=bool
            )
        elif 0 in attention_mask:
            padding_mask = attention_mask

        inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

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
