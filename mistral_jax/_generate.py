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
    k_th_value = jnp.take_along_axis(
        logits, sorted_indices[..., top_k - 1][..., None], axis=-1
    ).squeeze(-1)
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
        filter_value: the value used to replace the filtered entries
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


@functools.partial(jax.jit, static_argnames=("top_k", "top_p"))
def sample_with_tk_tp(rng, logits, top_k, top_p):
    return jax.random.categorical(rng,
                                  top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p))


@functools.partial(jax.jit, static_argnames=("length", "axis"))
def _pad_to(x, length, axis=0):
    pad_shape = x.shape[:axis] + (length - x.shape[axis],) + x.shape[axis + 1:]
    return jnp.concatenate((jnp.zeros(pad_shape), x), axis=axis)


def generate(
    params,
    eval_fn,
    prompt_tokens: list | jnp.ndarray,
    do_sample: bool = True,
    seed: int = 0,
    max_len: int = 100,
    top_k: int = 0,
    top_p: float = 0.0,
    temp: float = 1.0,
    generation_only: bool = False,
):
    """
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt_tokens: the tokenized prompt
        do_sample: whether to sample the distribution or take the argmax
        seed: random seed
        max_len: the max generation length
        top_k: top k
        top_p: top p
        temp: temperature
        generation_only: return newly generated tokens only
    Returns:
        the completed token array (containing the prompt)
    """
    if isinstance(prompt_tokens, list):
        current_state = jnp.array(prompt_tokens)
    elif len(prompt_tokens.shape) == 1:
        current_state = prompt_tokens[None, :]
    else:
        current_state = prompt_tokens

    rng = jax.random.PRNGKey(seed)

    if do_sample:
        sample_fn = sample_with_tk_tp
    else:
        sample_fn = lambda rng, logits, *args: jnp.argmax(logits, axis=-1)

    first_generated_logit, past_key_values = eval_fn(
            params, current_state, past_key_values=None, use_cache=True
    )
    first_generated_tok = sample_fn(rng, first_generated_logit[:, -1:] * 1.0 / temp, top_k, top_p)
    past_key_values = jax.tree_map(functools.partial(_pad_to, length=max_len, axis=2), past_key_values)

    @jax.jit
    def loop_fn(past_kv_and_rng_and_out, i):
        past_key_values, key, tok = past_kv_and_rng_and_out
        key, subkey = jax.random.split(key)
        outputs, past_key_values = eval_fn(
            params, tok, past_key_values=past_key_values, use_cache=True, unpadded_past_kv_length=i
        )

        logits = outputs[:, -1:] * 1.0 / temp

        out_tk = sample_fn(rng, logits, top_k, top_p)

        return (past_key_values, key, out_tk), out_tk.squeeze(1).T

    generated_toks = jax.lax.scan(loop_fn,
                                  (past_key_values, rng, current_state[:, -1:]),
                                  jnp.arange(1, max_len),
                                  )[1].T

    if generation_only:
        return jnp.concatenate((first_generated_tok, generated_toks), axis=-1)
    else:
        return jnp.concatenate((current_state, first_generated_tok, generated_toks), axis=-1)
