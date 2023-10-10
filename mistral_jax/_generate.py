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

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("top_k", "filter_value"))
def top_k_filtering(logits, top_k=32, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    sorted_indices = jnp.argsort(-logits)
    k_th_value = jnp.take_along_axis(logits, sorted_indices[..., top_k - 1][..., None], axis=-1).squeeze(-1)
    logits = jnp.where(logits < k_th_value[..., None], filter_value, logits)

    return logits


@functools.partial(jax.jit, static_argnames=("top_p", "filter_value"))
def top_p_filtering(logits, top_p=0.9, filter_value=-float("Inf")):
    sorted_indices = jnp.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs[:-1] > top_p

    indices_to_remove = sorted_indices[sorted_indices_to_remove + 1]
    logits = logits.at[indices_to_remove].set(filter_value)
    return logits


@functools.partial(jax.jit, static_argnames=("top_k", "top_p", "filter_value"))
def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
):
    """
    Args:
        logits: original logits
        top_k: keep only top k tokens with the rest marked as 'filter_value'
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            The rest are marked as 'filter_value'.
    """
    logits = jax.lax.cond(
        top_k > 0,
        lambda x: top_k_filtering(x, top_k=top_k, filter_value=filter_value),
        lambda x: x,
        logits,
    )
    logits = jax.lax.cond(
        top_p > 0.0,
        lambda x: top_p_filtering(x, top_p=top_p, filter_value=filter_value),
        lambda x: x,
        logits,
    )

    return logits


# TODO: generation is super slow because jit compiles every single shape of kv cache...... Need a stride to avoid this.
def generate(
    params,
    eval_fn,
    prompt_tokens: list | jnp.ndarray,
    seed: int = 0,
    max_len: int = 100,
    top_k: int = 0,
    top_p: float = 0.0,
    temp: float = 1.0,
    caching_stride: int = 16,
):
    """
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt_tokens: the tokenized prompt
        seed: random seed
        max_len: the max generation length
        top_k: top k
        top_p: top p
        temp: temperature
        caching_stride: TODO: the stride for bumping the shape of the sequence length axis.
            Larger stride avoids compiling too many functions.
    Returns:
        the completed token array (containing the prompt)
    """
    if isinstance(prompt_tokens, list):
        current_state = jnp.array(prompt_tokens)
    elif len(prompt_tokens.shape) == 1:
        current_state = prompt_tokens[None, :]
    else:
        current_state = prompt_tokens

    past_key_values = None
    rng = jax.random.PRNGKey(seed)
    for _ in range(max_len):
        key, subkey = jax.random.split(rng)
        if past_key_values is None:
            tok = current_state
        else:
            tok = current_state[:, -1:]
        outputs, past_key_values = eval_fn(
            params, tok, past_key_values=past_key_values, use_cache=True
        )

        logits = outputs[:, -1:] * 1.0 / temp
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        out_tk = jax.random.categorical(subkey, logits)

        current_state = jnp.concatenate((current_state, out_tk), axis=-1)

    return current_state
