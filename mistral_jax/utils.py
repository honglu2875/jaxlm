import jax
import jax.numpy as jnp
import torch


def torch_to_jax_states(input: torch.nn.Module | dict):
    """
    Converts the states of a PyTorch model to JAX states.
    """
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
        if len(split) > 1 and split[1].isdigit():
            levels = [split[0] + "_" + split[1], *split[2:]]
        else:
            levels = split

        if levels[-2] in _exclude_keys:
            _key_map = {}
        else:
            _key_map = _emb_key_map if levels[0] == "embed_tokens" else _dense_key_map

        if levels[-1] in _key_map:
            levels[-1], func = _key_map[levels[-1]]
            val = func(v.numpy())
        else:
            val = v.numpy()

        _dict = jax_states["params"]
        for i, l in enumerate(levels):
            _dict[l] = _dict.setdefault(l, {} if i < len(levels) - 1 else val)
            _dict = _dict[l]

    return jax_states
