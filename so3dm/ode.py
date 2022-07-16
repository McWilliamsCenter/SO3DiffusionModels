# Implement simple ODE solver on SO(3)
import jax
import jax.numpy as jnp
from jaxlie import SO3

def geomodeint(func, X0, t, method='Heun'):
    """ Geometric ODE solver using Heun's method.

    Reference for ODE solver on manifold:
    https://hal.archives-ouvertes.fr/hal-01328729/document

    Args:
        func: function to be solved.
        X0: initial condition as a quaternion.
        t: time vector.
    Returns:
        Values of the solution X at times t.
    """
    def scan_func_heun(carry, next_t):
        # Get the time step
        t = carry[0]
        h = next_t - t
        X = carry[1]
        F1 = h * func(X, t)
        u = jax.vmap(lambda f1,x: (SO3(x) @ SO3.exp(0.5*f1)).wxyz)(F1,X)
        F2 = h * func(u, t + 0.5*h)
        X = jax.vmap(lambda f2,x: (SO3(x) @ SO3.exp(f2)).wxyz)(F2,X)
        return [next_t, X], X

    def scan_func_euler(carry, next_t):
        # Get the time step
        t = carry[0]
        h = next_t - t
        X = carry[1]
        F1 = h * func(X, t)
        X = jax.vmap(lambda f,x: (SO3(x) @ SO3.exp(f)).wxyz)(F1,X)
        return [next_t, X], X

    if method == 'Heun':
        _, Xs = jax.lax.scan(scan_func_heun, [t[0], X0], t[1:])
    elif method == 'Euler':
        _, Xs = jax.lax.scan(scan_func_euler, [t[0], X0], t[1:])
    else:
        raise ValueError('Method not supported.')

    return Xs