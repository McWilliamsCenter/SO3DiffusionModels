import jax
import jax.numpy as jnp


def q_exp(q):
    """Returns the quaternion exponential of a quaternion.

    Args:
      q: A quaternion of shape [..., 4].
    Returns:
      The quaternion exponential of q.
    """
    a = q[..., 0]
    v = q[..., 1:]
    exp_scalar = jnp.exp(a)
    v_norm = jnp.linalg.norm(v, axis=-1)
    cos_v = jnp.cos(v_norm)
    sin_v = jnp.sin(v_norm)
    sin_v_over_v_norm = sin_v / v_norm
    scalar_res = exp_scalar*cos_v
    vec_res = sin_v_over_v_norm*exp_scalar*v
    return jnp.concatenate((scalar_res[..., jnp.newaxis], vec_res), axis=-1)
