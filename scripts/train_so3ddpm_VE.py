# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import sys
import os
sys.path.append('../')

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
from so3dm.metrics import c2st
import pickle

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/so3ddpm_VE/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 400_000  , "Total number of training steps.")
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

 
    
flags.DEFINE_integer("n_steps", 500, "Number of steps in noise schedule")
flags.DEFINE_float("start_noise", 0.0001, "The value in which to start the noise schedule")
flags.DEFINE_float("stop_noise", 0.2, "The value in which to stop the noise schedule")
 
flags.DEFINE_integer("n_folds", 10, "Number of folds in c2st")

 

 
    
FLAGS = flags.FLAGS







@jax.jit
def get_batch(batch, key, noise_schedule, delta ):
    key1, key2, key3 = jax.random.split(key,3)
    # Sample from the target distribution
    @jax.vmap
    def sample(q, scale, seed):
        key1, key2 = jax.random.split(seed)
        x = SO3(q) 

        # Sampling from current temperature
        dist = IsotropicGaussianSO3(x, scale)
        qn = dist.sample(seed=key1)

        # Sampling from next temperature step 
        dist2 = IsotropicGaussianSO3(qn, jnp.sqrt(delta))
        qnplus1 = dist2.sample(seed=key2)

        return {'x': x.wxyz, 'yn': qn, 'yn+1': qnplus1, 
                'sn':scale, 'sn+1':jnp.sqrt(scale**2 + delta)}  
    # Sample random noise levels from the schedule
    s = jax.random.choice(key3, noise_schedule, shape=[FLAGS.batch_size])
    s = jnp.sqrt(s)
    
    # Sample random rotations
    return sample(batch['pos_quat'], s, jax.random.split(key2, FLAGS.batch_size))







def model_fn(x,s):
    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256, 256, 256], activation=jax.nn.leaky_relu)(net)
    # We use a residual connection, because we expect deviations to be small
    mu = hk.Linear(4)(net) + x
    # We force the normalization of the quaternion
    mu = mu / jnp.linalg.norm(mu,axis=-1, keepdims=True)
    scale = jax.nn.softplus(hk.Linear(1)(net)) + 0.00001
    return mu, scale

 

def main(_):
    output_dir = FLAGS.output_dir 

    # Just to make sure jax is initialized before TF
    jnp.linalg.inv(jnp.eye(3))

    # Instantiate the network
    model = hk.without_apply_rng(hk.transform(model_fn))  
    
    rng_seq = hk.PRNGSequence(42)
    noise_schedule, delta = jnp.linspace(FLAGS.start_noise, FLAGS.stop_noise, 
                                            FLAGS.n_steps, retstep=True)   

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

        params = model.init(next(rng_seq), jnp.zeros([1,4]),jnp.zeros([1,1]))
         
        # Creating the optimizer
        optimizer = optax.chain(optax.adam(learning_rate=FLAGS.learning_rate))
        opt_state = optimizer.init(params)

        # Define the loss function
        def loss_fn(params, rng_key, batch):
            s = batch['sn+1'].reshape([-1,1])
            mu, scale = model.apply(params, batch['yn+1'], s)

            @jax.vmap
            def fn(x, mu, scale):
                dist = IsotropicGaussianSO3(mu, scale, 
                                            force_small_scale=True)
                return dist.log_prob(x)

            loss = - fn(batch['yn'], mu, scale)

            return jnp.mean(loss)

        @jax.jit
        def update(params, rng_key, opt_state, batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, rng_key, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return loss, new_params, new_opt_state

        summary_writer = tensorboard.SummaryWriter(output_dir)

        print('training begins')
        for step in tqdm(range(FLAGS.training_steps)):
            batch = get_batch(next(dset), next(rng_seq), noise_schedule, delta)
            
            loss, params, opt_state = update(params, next(rng_seq), opt_state, batch)
 
            if jnp.isnan(loss):
                break

            if step%50==0:
                summary_writer.scalar('train_loss', loss, step)
                #summary_writer.scalar('learning_rate', FLAGS.learning_rate*lr_schedule(step), step)

            if step%10000 ==0:
                with open(output_dir+ '/' + FLAGS.dataset + '_model-%d.pckl'%step, 'wb') as file:
                    pickle.dump(params, file)

        summary_writer.flush()

        with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'wb') as file:
            pickle.dump(params, file)

    with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'rb') as file:
        params = pickle.load(file)

     
 

    # Starting sampling from the trained model
    X0 = jax.vmap(lambda k: SO3.sample_uniform(k).wxyz)(jax.random.split(next(rng_seq), FLAGS.test_nsamples))
    @jax.jit
    @jax.vmap
    def fn_sample(mu,s,key):
        return IsotropicGaussianSO3(mu, s, force_small_scale=True).sample(seed=key)
    
    x_t = X0
    for variance in noise_schedule[::-1]:
        mu, s = model.apply(params, x_t, jnp.sqrt(variance)*jnp.ones([FLAGS.test_nsamples,1]))
        x_t = fn_sample(mu, s, jax.random.split(next(rng_seq), FLAGS.test_nsamples))
    
    
 
    with open(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
        onp.save(f, x_t)

    visualize_so3_density(jax.vmap(lambda q: SO3(q).as_matrix())(x_t), 100);
    plt.savefig(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + ".png")
    
    if FLAGS.compute_c2st:    
        true_samp_loc = 'reference_distribution/' + FLAGS.dataset + '_true_200_000.npy'



        with open(true_samp_loc , 'rb') as file:
            true_samp = onp.load(file)

        seed = 1
        if true_samp.shape[1] == 3:
            true_samp = jax.vmap(lambda m: SO3.from_matrix(m).wxyz )(true_samp) # print(X.shape)


        print("Calculating c2st ... ")
 
    
        c2_score = c2st(true_samp, x_t, seed, FLAGS.n_folds)

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
