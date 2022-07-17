import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from jaxlie import SO3

@jax.jit
def sample_checkerboard(seed, size=jnp.pi):
    """Samples one point from a checkerboard distribution on SO(3).
    Returns both rotation matrix and quaternion representation.
    """
    key1, key2, key3 = jax.random.split(seed, 3)
    x1 = jax.random.uniform(key=key1) * size - size/2
    x2_ = jax.random.uniform(key=key2) - jax.random.randint(minval=0, maxval=2,shape=[], key=key3) * 2
    x2 = x2_ + (jnp.floor(x1) % 2)
    data = jnp.stack([x1, x2]) * size/2
    rot = SO3.from_rpy_radians(pitch=data[0]/2, yaw=data[1], roll=0)
    # Fix positivity convention
    rot = SO3.exp(rot.log())
    return rot.as_matrix(), rot.wxyz

class Checkerboard(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for checkerboard dataset on SO(3)."""

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
      R, quat = sample_checkerboard(seed)
      yield i, {
          'pos_quat': quat,
          'pos_mat': R,
      }