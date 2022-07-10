import jax
import jax.numpy as jnp
import numpy as np
from jaxlie import SO3

import tensorflow_probability as tfp; tfp = tfp.substrates.jax
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import dtype_util

def _isotropic_gaussian_so3_small(omg, scale):
    """ Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L99
    This function implements the approximation of the density function of omega of the isotropic Gaussian distribution. 
    """
    eps = scale**2
    # TODO: check for stability and maybe replace by limit in 0 for small values
    small_number = 1e-9
    small_num = small_number / 2 
    small_dnm = (1-np.exp(-1. * np.pi**2 / eps)*(2  - 4 * (np.pi**2) / eps   )) * small_number

    return 0.5 * np.sqrt(jnp.pi) * (eps ** -1.5) * jnp.exp((eps - (omg**2 / eps))/4) / (jnp.sin(omg/2) + small_num)            \
        * ( small_dnm + omg - ((omg - 2*jnp.pi)*jnp.exp(jnp.pi * (omg - jnp.pi) / eps) + (omg + 2*jnp.pi)*jnp.exp( -jnp.pi * (omg+jnp.pi) / eps) ))            

def _isotropic_gaussian_so3(omg, scale, lmax = None):
    """ Borrowed from: https://github.com/tomato1mule/edf/blob/1dd342e849fcb34d3eb4b6ad2245819abbd6c812/edf/dist.py#L82
    This function implements the density function of omega of the isotropic Gaussian distribution. 
    """
    eps = scale**2

    if lmax is None:
        lmax = max(int( 3. / np.sqrt(eps)) , 2)

    small_number = 1e-9
    sum = 0.
    # TODO: replace by a scan
    for l in range(lmax + 1):
        sum = sum + (2*l+1)    *    np.exp(-l*(l+1) * eps)    *    (  jnp.sin((l+0.5)*omg) + (l+0.5)*small_number  )    /    (  jnp.sin(omg/2) + 0.5*small_number  )
    return sum


class IsotropicGaussianSO3(tfp.distributions.Distribution):
    
    def __init__(self, loc, scale,
               validate_args=False,
               allow_nan_stats=True,
               name='IsotropicGaussianSO3'):
        parameters = dict(locals())
        dtype = dtype_util.common_dtype([loc, scale])
        
        if not isinstance(loc, SO3):
            loc = SO3(loc)
        self._loc = loc
        self._scale = scale
        
        # Precomputing array of values for doing inverse  cdf sampling
        self._x = jnp.linspace(0, jnp.pi, 1024)
        y = (1 - jnp.cos(self._x))/ jnp.pi * self._f(self._x)
        y = jnp.cumsum(y) * jnp.pi / 1024
        self._y = y / y.max()
        
        super(IsotropicGaussianSO3, self).__init__(
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name,
          dtype=dtype)
        
    def _f(self, angles):
        if self._scale < 1:
            prob = _isotropic_gaussian_so3_small(angles, self._scale/jnp.sqrt(2))
        else:
            prob = _isotropic_gaussian_so3(angles, self._scale/jnp.sqrt(2), lmax=3)
        return prob
        
    def _log_prob(self, q):
        if len(q.shape) == 1:
            q = jnp.expand_dims(q, axis=0)
        @jax.vmap
        def get_angles(x):
            axis_angle = (self._loc.inverse() @ SO3(x)).log()
            return jnp.linalg.norm(axis_angle, axis=-1)    
        angles = get_angles(q)
        return jnp.log(self._f(angles))
    
    def _sample_n(self, n, seed=None):
        key1, key2 = jax.random.split(seed)
        rand_angle = jax.random.uniform(shape=[n], key=key1)
        angle = jnp.interp(rand_angle, self._y, self._x)
        axis = jax.random.normal(shape=[n,3], key=key2)
        axis = axis / jnp.linalg.norm(axis, axis=-1, keepdims=True)
        aa = angle[..., jnp.newaxis] * axis
        return jax.vmap(lambda axis_angle: (self._loc @ SO3.exp(axis_angle)).wxyz)(aa)