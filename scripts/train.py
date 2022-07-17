# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard
import haiku as hk
import optax

import jax
import jax.numpy as jnp
from jaxlie import SO3
from so3dm.distributions import IsotropicGaussianSO3

import pickle

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/run1", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 512, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initiatl learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 100000, "Total number of training steps.")

FLAGS = flags.FLAGS

def lr_schedule(step):
  """Step learning rate schedule rule."""
  lr = (1.0 * FLAGS.batch_size) / 512
  boundaries = jnp.array((0.2, 0.6) ) * FLAGS.training_steps
  values = jnp.array([1., 0.1, 0.01]) * lr
  index = jnp.sum(boundaries < step)
  return jnp.take(values, index)


@jax.jit
def get_batch(batch, key, noise_dist_std=0.8):
    key1, key2 =jax.random.split(key)
    # Sample random noise from target noise distribution
    s = noise_dist_std * jnp.abs(jax.random.normal(shape=[FLAGS.batch_size], key=key1)) + 1e-2
    @jax.vmap
    def sample(q, scale, seed):
        x = SO3(q)
        dist = IsotropicGaussianSO3(x, scale)
        qn = dist.sample(seed=seed)
        
        def fn(s,q):
            return dist.log_prob( (SO3(q) @ SO3.exp(s)).wxyz)
        score = jax.grad(fn)(jnp.zeros(3), qn)
        
        return {'x': x.wxyz, 'y': qn, 'score': score, 's':scale.reshape([1])}
    # Generates training set batch
    return sample(batch['pos_quat'], s, jax.random.split(key2, FLAGS.batch_size))

def model_fn(x,s):
    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256], activation=jax.nn.leaky_relu)(net)
    net = hk.Linear(3)(net)
    return net/s

def main(_):
    # Just to make sure jax is initialized before TF
    jnp.linalg.inv(jnp.eye(3))

    # Open the dataset
    dset = tfds.load(FLAGS.dataset, split="train")
    dset = dset.repeat()
    dset = dset.shuffle(buffer_size=10000)
    dset = dset.batch(FLAGS.batch_size)
    dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dset = dset.as_numpy_iterator()
    t = next(dset)
    
    # Instantiate the network
    model = hk.without_apply_rng(hk.transform(model_fn))

    rng_seq = hk.PRNGSequence(42)

    # Initialize weights
    params = model.init(next(rng_seq),
                               jnp.zeros((1, 4)),
                               jnp.ones((1, 1)))

    # Creating the optimizer
    optimizer = optax.chain(
        optax.adam(learning_rate=FLAGS.learning_rate),
        optax.scale_by_schedule(lr_schedule)
    )
    opt_state = optimizer.init(params)

    # Define the loss function
    def loss_fn(params, batch):
        score_pred = model.apply(params, batch['y'], batch['s'])
        loss = (batch['s'].squeeze()*jnp.linalg.norm(score_pred - batch['score'], axis=-1))**2
        return jnp.mean(loss)

    @jax.jit
    def update(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    summary_writer = tensorboard.SummaryWriter(FLAGS.output_dir)

    print('training begins')
    for step in range(FLAGS.training_steps):
        batch = get_batch(next(dset), next(rng_seq))
        # Sampling another batch if the current one had a NaN
        if jnp.isnan(batch['score']).any():
            batch = get_batch(next(dset), next(rng_seq))
        
        loss, params, opt_state = update(params, opt_state, batch)

        if step%50==0:
            summary_writer.scalar('train_loss', loss, step)
            summary_writer.scalar('learning_rate', FLAGS.learning_rate*lr_schedule(step), step)
            print(step, loss)

        if step%10000 ==0:
            with open(FLAGS.output_dir+'/model-%d.pckl'%step, 'wb') as file:
                pickle.dump(params, file)

    summary_writer.flush()

    with open(FLAGS.output_dir+'/model-final.pckl', 'wb') as file:
        pickle.dump(params, file)

if __name__ == "__main__":
    app.run(main)