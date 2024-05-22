from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax.numpy as jnp

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


