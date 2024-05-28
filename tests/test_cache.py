import jax.numpy as jnp
from jaxlm.cache import KVCache
import jax


def test_cache():
    seq_axis = 1
   
    shape = (4, 16, 32, 128)
    cache = KVCache.init(shape)
    assert cache.k.shape == shape
    assert cache.v.shape == shape
    assert cache.end_pos == 0

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    new_k = jax.random.uniform(k1, (4, 1, 32, 128))
    new_v = jax.random.uniform(k2, (4, 1, 32, 128))

    cache = cache.update(new_k, new_v)

    assert jnp.allclose(cache.k, jnp.concatenate([new_k, jnp.zeros((4, 15, 32, 128))], axis=1))
    assert jnp.allclose(cache.v, jnp.concatenate([new_v, jnp.zeros((4, 15, 32, 128))], axis=1))
    assert cache.end_pos == 1

    cache = cache.update(new_k, new_v)

    assert jnp.allclose(cache.k, jnp.concatenate([new_k, new_k, jnp.zeros((4, 14, 32, 128))], axis=1))
    assert jnp.allclose(cache.v, jnp.concatenate([new_v, new_v, jnp.zeros((4, 14, 32, 128))], axis=1))
    assert jnp.all(cache.end_pos == jnp.array([2], dtype=jnp.int32))

    
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    new_k2 = jax.random.uniform(k1, (4, 2, 32, 128))
    new_v2 = jax.random.uniform(k2, (4, 2, 32, 128))

    cache = cache.update(new_k2, new_v2)

    assert jnp.allclose(cache.k, jnp.concatenate([new_k, new_k, new_k2, jnp.zeros((4, 12, 32, 128))], axis=1))
    assert jnp.allclose(cache.v, jnp.concatenate([new_v, new_v, new_v2, jnp.zeros((4, 12, 32, 128))], axis=1))
    assert jnp.all(cache.end_pos == jnp.array([4], dtype=jnp.int32))
