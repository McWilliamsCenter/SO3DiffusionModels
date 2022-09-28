# Script used to train and evaluate models from the 
# Riemannian SGM package
from absl import app
from absl import flags

import os
from hydra import initialize, compose

os.environ['GEOMSTATS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["WANDB_START_METHOD"] = "thread"


import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

from flax.metrics import tensorboard
from jaxlie import SO3
from so3dm.distributions.isotropic_gaussian import IsotropicGaussianSO3
from so3dm.plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt
from collections import namedtuple
from so3dm.metrics import c2st

from hydra.utils import instantiate, get_class

from score_sde.losses import get_ema_loss_step_fn
from tqdm import tqdm
import pickle

import optax
import haiku as hk

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("model", "RSGM", "Type of model to sure, can be 'RSGM' or 'Moser'")
flags.DEFINE_string("output_dir", "models/rsgm/", "Folder where to store model and training info.")
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")

#Metric
flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")
flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")

FLAGS = flags.FLAGS

TrainState = namedtuple(
    "TrainState",
    [
        "opt_state",
        "model_state",
        "step",
        "params",
        "ema_rate",
        "params_ema",
        "rng",
    ],
)

def main(_):
  output_dir = FLAGS.output_dir + '/' + FLAGS.model + '/'

  # Load the configuration
  with initialize(config_path="riemannian-score-sde/config", ):
    if FLAGS.model == 'RSGM':
      cfg = compose("main", overrides=["experiment=so3", "model=rsgm", 
                              "dataset.K=16", "loss=dsmv", "generator=lie_algebra", "optim.learning_rate=5e-4",
                              "seed=0"])
    elif FLAGS.model == 'Moser':
      cfg = compose("main", overrides=["experiment=so3", "model=moser", 
                              "dataset.K=16", "generator=lie_algebra", "optim.learning_rate=5e-4", "loss.K=1000",
                              "loss.alpha_m=10", "seed=0"])
    else:
      raise NotImplementedError
    print(cfg)
  rng_seq = hk.PRNGSequence(cfg.seed) # which is a bit dumb, it will always be 0

  data_manifold = instantiate(cfg.manifold)
  transform = instantiate(cfg.transform, data_manifold)
  model_manifold = transform.domain
  flow = instantiate(cfg.flow, manifold=model_manifold)
  base = instantiate(cfg.base, model_manifold, flow)
  pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

  
  # Define model function
  def model(y, t, context=None):
    """Vector field s_\theta: y, t, context -> T_y M"""
    output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
    score = instantiate(
        cfg.generator,
        cfg.architecture,
        cfg.embedding,
        output_shape,
        manifold=model_manifold,
    )
    # TODO: parse context into embedding map
    if context is not None:
        t_expanded = jnp.expand_dims(t.reshape(-1), -1)
        if context.shape[0] != y.shape[0]:
            context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
        context = jnp.concatenate([t_expanded, context], axis=-1)
    else:
        context = t
    return score(y, context)
  model = hk.transform_with_state(model)

  if FLAGS.train:
    # Prepare dataset
    dset = tfds.load(FLAGS.dataset, split="train")
    dset = dset.repeat()
    dset = dset.shuffle(buffer_size=10000)
    dset = dset.map(lambda x: (x['pos_mat'], jnp.atleast_1d(0)))
    dset = dset.batch(cfg.batch_size)
    dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_ds = dset.as_numpy_iterator()

    # Initialize model parameters
    t = jnp.zeros((cfg.batch_size, 1))
    data, context = next(train_ds)
    params, state = model.init(rng=next(rng_seq), 
                              y=transform.inv(data), 
                              t=t, 
                              context=None)

    # Prepare optimizer
    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
    opt_state = optimiser.init(params)
    train_state = TrainState(
      opt_state=opt_state,
      model_state=state,
      step=0,
      params=params,
      ema_rate=cfg.ema_rate,
      params_ema=params,
      rng=next(rng_seq), 
    )

    loss = instantiate(
        cfg.loss, pushforward=pushforward, model=model, eps=cfg.eps, train=True
    )
    train_step_fn = get_ema_loss_step_fn(loss, optimizer=optimiser, train=True)
    train_step_fn = jax.jit(train_step_fn)

    summary_writer = tensorboard.SummaryWriter(output_dir)

    print('training begins')
    for step in tqdm(range(cfg.steps)):
        data, context = next(train_ds)
        batch = {"data": data, "context": None}
        (rng, train_state), loss = train_step_fn((next(rng_seq), train_state), batch)
        if jnp.isnan(loss).any():
            print("Oh no, a NaN :-( ")
            break

        if step%50==0:
            summary_writer.scalar('train_loss', loss, step)

    summary_writer.flush()

    with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'wb') as file:
       pickle.dump((params, train_state), file)

  with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'rb') as file:
    params, train_state = pickle.load(file)

  # Loading testing dataset
  dset = tfds.load(FLAGS.dataset, split="test")
  dset = dset.map(lambda x: (x['pos_mat'], jnp.atleast_1d(0)))
  dset = dset.batch(cfg.batch_size)
  dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_ds = dset.as_numpy_iterator()

  # Starting sampling from the trained model
  model_w_dicts = (model, train_state.params_ema, train_state.model_state)
  if FLAGS.model == 'RSGM':
    sampler_kwargs = dict(N=100, eps=cfg.eps, predictor="GRW")
  else:
    sampler_kwargs = {}
  sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)
  x0, context = next(test_ds)
  shape = (int(cfg.batch_size), *transform.inv(x0).shape[1:])

  samples = []
  for i in range(FLAGS.test_nsamples//cfg.batch_size + 1):
    x = sampler(next(rng_seq), shape, context[0])
    samples.append(x)
  samples = jnp.stack(samples, axis=0)[:FLAGS.test_nsamples]
    
  with open(output_dir + FLAGS.dataset + '_' + str(len(x)) + ".npy", "wb") as f:
    # Converting to quaternions
    q = jax.vmap(lambda R: SO3.from_matrix(R).wxyz)(x)
    onp.save(f, q)

  visualize_so3_density(x, 100);
  plt.savefig(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + ".png")
  
  if FLAGS.compute_c2st:    
        true_samp_loc = 'reference_distribution/' + FLAGS.dataset + '_true_200_000.npy'

        with open(true_samp_loc , 'rb') as file:
            true_samp = onp.load(file)

        seed = 1
        if true_samp.shape[1] == 3:
            true_samp = jax.vmap(lambda m: SO3.from_matrix(m).wxyz )(true_samp) # print(X.shape)

        print("Calculating c2st ... ")
        c2_score = c2st(true_samp, q, seed, FLAGS.n_folds)

        with open(output_dir+"output.txt", "a") as f:
          print( "C2ST score: "+ str(c2_score), file=f)

        print(true_samp.shape[1])
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")

        print("C2ST score: "+ str(c2_score))
        
if __name__ == "__main__":
  app.run(main)
