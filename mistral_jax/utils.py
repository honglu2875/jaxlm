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

import jax.numpy as jnp
import torch
import orbax


def torch_to_jax_states(
    input: torch.nn.Module | dict, dtype: str | torch.dtype = "bf16"
):
    """
    Converts the states of a PyTorch model to JAX states.
    """
    _to_jnp_dtype = {
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
        "bf16": jnp.bfloat16,
    }

    if isinstance(input, torch.nn.Module):
        states = input.state_dict()
    elif isinstance(input, dict):
        states = input
    else:
        raise TypeError(
            f"Expected input to be either a PyTorch module or a dict, got {type(input)}."
        )

    jax_states = {"params": {}}

    _dense_key_map = {"weight": ("kernel", lambda x: x.T)}
    _emb_key_map = {"weight": ("embedding", lambda x: x)}
    _exclude_keys = {"post_attention_layernorm", "input_layernorm", "norm"}

    for k, v in states.items():
        if k.endswith("bias"):
            raise NotImplementedError(
                "Not implemented for bias conversion as Mistral does not use bias."
            )
        split = k.split(".")
        for i, s in enumerate(split):
            if s.isdigit():
                split[i - 1] += "_" + s
                split.pop(i)

        if split[-2] in _exclude_keys:
            _key_map = {}
        else:
            _key_map = _emb_key_map if "embed_tokens" in split else _dense_key_map

        if split[-1] in _key_map:
            split[-1], func = _key_map[split[-1]]
            val = func(v.numpy().astype(_to_jnp_dtype[dtype]))
        else:
            val = v.numpy().astype(_to_jnp_dtype[dtype])

        _dict = jax_states["params"]
        for i, l in enumerate(split):
            _dict[l] = _dict.setdefault(l, {} if i < len(split) - 1 else val)
            _dict = _dict[l]

    return jax_states


def save(params, path='tmp/'):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(path, params)


def load(path='tmp/', item=None):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(path, item=item)
