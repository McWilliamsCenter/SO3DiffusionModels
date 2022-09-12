import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from jaxlie import SO3

@jax.jit
def sample_4gauss(seed ):
    """Samples one point from a 4 gaussian blob distribution on SO(3).
    Returns both rotation matrix and quaternion representation.
    """
    key1, key2, key3 = jax.random.split(seed, 3)
    
    sigma = 1/6
    center = 0.6
    center2 = 0.5
    chance = jax.random.choice(key3, jnp.array([0,1,2,3]))
    set0 = jnp.stack([jax.random.normal(key=key1)*sigma - center, jax.random.normal(key=key2)*sigma-center2, 0  ])
    set1 = jnp.stack([jax.random.normal(key=key1)*sigma + center, jax.random.normal(key=key2)*sigma-center2, 0  ])
    set2 = jnp.stack([jax.random.normal(key=key1)*sigma + (-jnp.pi-center), jax.random.normal(key=key2)*sigma+center2, jnp.pi  ])
    set3 = jnp.stack([jax.random.normal(key=key1)*sigma + (-jnp.pi+center), jax.random.normal(key=key2)*sigma+center2, jnp.pi  ])
 
    
    data = jnp.stack([set0, set1, set2, set3]) 
    chosen_data = data[chance]
    rot = SO3.from_rpy_radians(pitch=chosen_data[0], yaw=chosen_data[1], roll=chosen_data[2] )
    
    # Fix positivity convention
    rot = SO3.exp(rot.log())
    return rot.as_matrix(), rot.wxyz

class Gaussian_blob_4(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for 4 Gaussian blob dataset on SO(3)."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'pos_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
            'pos_mat': tfds.features.Tensor(shape=(3,3), dtype=tf.float32),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    key = jax.random.PRNGKey(0)
    key_train, key_test = jax.random.split(key, 2)
    return {
        'train': self._generate_examples(key_train),
        'test': self._generate_examples(key_test),
    }

  def _generate_examples(self, key):
    """Generator of examples for each split."""
    for i in range(200000):
      key, seed = jax.random.split(key)
      R, quat = sample_4gauss(seed)
      yield i, {
          'pos_quat': quat,
          'pos_mat': R,
      }