import jax
import jax.numpy as jnp


def q_exp(q):
    """Returns the quaternion exponential of a quaternion.

    Args:
      q: A quaternion.
    Returns:
      The quaternion exponential of q.
    """
    a = q[:, 0]
    v = q[:, 1:]
    exp_scalar = jnp.exp(a)
    v_norm = jnp.linalg.norm(v, axis=1)
    v_norm = v_norm.T
    cos_v = jnp.cos(v_norm)
    sin_v = jnp.sin(v_norm)
    sin_v_over_v_norm = jnp.divide(sin_v, v_norm)
    scalar_res = exp_scalar*cos_v
    vec_res = sin_v_over_v_norm[:, jnp.newaxis]*exp_scalar[:, jnp.newaxis]*v
    return jnp.column_stack((scalar_res, vec_res))
