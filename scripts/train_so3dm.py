# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard
import haiku as hk
import optax

from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as onp
from jaxlie import SO3
from so3dm.distributions import IsotropicGaussianSO3
from so3dm.ode import geomodeint
from so3dm.plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt

import pickle

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/score_matching", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 512, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 400000, "Total number of training steps.")
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 20000, "Number of samples to draw at testing time.")

FLAGS = flags.FLAGS

def lr_schedule(step):
  """Step learning rate schedule rule."""
  lr = (1.0 * FLAGS.batch_size) / 512
  boundaries = jnp.array((0.2, 0.7) ) * FLAGS.training_steps
  values = jnp.array([1., 0.1, 0.01]) * lr
  index = jnp.sum(boundaries < step)
  return jnp.take(values, index)


@jax.jit
def get_batch(batch, key, noise_dist_std=1.2):
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
    x = jax.vmap(lambda u: SO3(u).log())(x)
    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256, 256], activation=jax.nn.silu)(net)
    net = hk.Linear(3)(net)
    return net/s

def main(_):
    output_dir = FLAGS.output_dir+"_"+FLAGS.dataset

    # Just to make sure jax is initialized before TF
    jnp.linalg.inv(jnp.eye(3))

    # Instantiate the network
    model = hk.without_apply_rng(hk.transform(model_fn))

    rng_seq = hk.PRNGSequence(42)

    if FLAGS.train:

        # Open the dataset
        dset = tfds.load(FLAGS.dataset, split="train")
        dset = dset.repeat()
        dset = dset.shuffle(buffer_size=10000)
        dset = dset.batch(FLAGS.batch_size)
        dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dset = dset.as_numpy_iterator()
        _ = next(dset)
        
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

        summary_writer = tensorboard.SummaryWriter(output_dir)

        print('training begins')
        for step in tqdm(range(FLAGS.training_steps)):
            batch = get_batch(next(dset), next(rng_seq))
            # Sampling another batch if the current one had a NaN
            if jnp.isnan(batch['score']).any():
                batch = get_batch(next(dset), next(rng_seq))
            
            loss, params, opt_state = update(params, opt_state, batch)

            if step%50==0:
                summary_writer.scalar('train_loss', loss, step)
                summary_writer.scalar('learning_rate', FLAGS.learning_rate*lr_schedule(step), step)

            if step%10000 ==0:
                with open(output_dir+'/model-%d.pckl'%step, 'wb') as file:
                    pickle.dump(params, file)

        summary_writer.flush()

        with open(output_dir+'/model-final.pckl', 'wb') as file:
            pickle.dump(params, file)

    with open(output_dir+'/model-final.pckl', 'rb') as file:
        params = pickle.load(file)

    # Starting sampling from the trained model
    X0 = jax.vmap(lambda k: SO3.sample_uniform(k).wxyz)(jax.random.split(next(rng_seq), FLAGS.test_nsamples))

    t0 = 10.
    @jax.jit
    def dynamics(x, t):
        return - 0.5*model.apply(params,x,jnp.ones([FLAGS.test_nsamples,1])*jnp.sqrt(t))

    ts = jnp.linspace(t0, 0.0, 2048)
    Y = geomodeint(dynamics, X0, ts)
    with open(output_dir+"/Rsamples.npy", "wb") as f:
        onp.save(f, Y[-1])

    visualize_so3_density(jax.vmap(lambda q: SO3(q).as_matrix())(Y[-1]),32);
    plt.savefig(output_dir+"/Rsamples.png")

if __name__ == "__main__":
    app.run(main)