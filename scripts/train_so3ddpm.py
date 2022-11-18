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
from so3dm.plotting import visualize_so3_density, visualize_so3_probabilities
import matplotlib.pyplot as plt

import pickle

from so3dm.metrics import c2st

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/so3ddpm_vexp/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.")
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")
flags.DEFINE_string("input_rotation_param", "axis-angle", "Parameterisation of the rotation at the input of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("output_rotation_param", "axis-angle", "Parameterisation of the rotation at the output of the NN either 'axis-angle' or 'matrix'")
flags.DEFINE_string("diffusion_type", "vexp", "Variance preserving or variance exploding diffusion 'vexp' or 'vpres'") 

flags.DEFINE_bool("compute_c2st", True, "Whether to compute the c2st score agianst the true samples")

flags.DEFINE_integer("n_steps", 128, "Number of steps in noise schedule")

flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")
    
FLAGS = flags.FLAGS

def lr_schedule(step):
  """Step learning rate schedule rule."""
  lr = (1.0 * FLAGS.batch_size) / 1024
  boundaries = jnp.array((0.2, 0.7) ) * FLAGS.training_steps
  values = jnp.array([1., 0.1, 0.01]) * lr
  index = jnp.sum(boundaries < step)
  return jnp.take(values, index)

def quat_power(quat, a):
    quat = SO3(quat)
    return SO3.exp(quat.log()*a).wxyz

@jax.jit
def get_batch(batch, key, noise_schedule):
    key1, key2 = jax.random.split(key,2)

    @jax.vmap
    def sample_vexp(q, temperature, seed):
        scale = noise_schedule[temperature]
        scalenplus1 = noise_schedule[temperature+1]
        delta = scalenplus1**2 - scale**2
        key1, key2 = jax.random.split(seed)
        q = SO3(q).wxyz
        # Sampling from current temperature
        dist = IsotropicGaussianSO3(q, scale)
        qn = dist.sample(seed=key1)

        # Sampling from next temperature step 
        dist2 = IsotropicGaussianSO3(qn, jnp.sqrt(delta))
        qnplus1 = dist2.sample(seed=key2)

        return {'x': q, 'yn': qn, 'yn+1': qnplus1, 
                'sn':scale, 'sn+1':scalenplus1}  

    @jax.vmap
    def sample_vpres(quaternion, temperature, seed):
        key1, key2 = jax.random.split(seed)
        alpha = noise_schedule[temperature]
        beta  = 1 - (noise_schedule[temperature+1] / alpha)
        quaternion = SO3(quaternion).wxyz
        
        # Sampling from the current temperature
        dist = IsotropicGaussianSO3(quat_power(quaternion, jnp.sqrt(alpha)), jnp.sqrt(1 - alpha))
        qn = dist.sample(seed=key1)

        # Sampling from the next temperature step
        dist2 = IsotropicGaussianSO3(quat_power(qn, jnp.sqrt(1 - beta)), jnp.sqrt(beta))
        qnplus1 = dist2.sample(seed=key2)
    
        return {'x': quaternion, 'yn': qn, 'yn+1': qnplus1, 
                'sn':alpha, 'sn+1':noise_schedule[temperature+1]}  

    # Sample random noise levels from the schedule
    temp_list =  jnp.arange(len(noise_schedule)-1, dtype='int32')
    temperature = jax.random.choice(key1, temp_list, shape=[FLAGS.batch_size])

    if FLAGS.diffusion_type == "vexp":
        sample = sample_vexp
    elif FLAGS.diffusion_type == "vpres":
        sample = sample_vpres
    else:
        raise NotImplementedError

    # Sample random rotations
    return sample(batch['pos_quat'], temperature, jax.random.split(key2, FLAGS.batch_size))

 
def model_fn(x,s):
    if FLAGS.input_rotation_param == 'axis-angle':
        x = jax.vmap(lambda u: SO3(u).log())(x)
    elif FLAGS.input_rotation_param == 'matrix':
        x = jax.vmap(lambda u: SO3(u).as_matrix().flatten())(x)
    else:
        raise NotImplementedError

    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256, 256, 256], activation=jax.nn.leaky_relu)(net)

    # Note that we only output the residual with respect to the input rotation
    if FLAGS.output_rotation_param == 'axis-angle':
        delta_mu = hk.Linear(3)(net) # axis-angle output
        delta_mu = jax.vmap(lambda u: SO3.exp(u).wxyz)(delta_mu)
    elif FLAGS.output_rotation_param == 'matrix':
        net = hk.Linear(6)(net)
        R1 = net[:,0:3] / jnp.linalg.norm(net[:,0:3], axis=-1, keepdims=True)
        R3 = jnp.cross(R1, net[:, 3:],axis=-1)
        R3 = R3 / jnp.linalg.norm(R3, axis=-1, keepdims=True)
        R2 = jnp.cross(R3, R1)
        delta_mu = jnp.stack([R1, R2, R3], axis=-1) # rotation matrix
        delta_mu = jax.vmap(lambda u: SO3.from_matrix(u).wxyz)(delta_mu)
    else:
        raise NotImplementedError

    scale = jax.nn.softplus(hk.Linear(1)(net)) + 0.0001
    return delta_mu, scale

def main(_):
    output_dir = FLAGS.output_dir 

    # Just to make sure jax is initialized before TF
    jnp.linalg.inv(jnp.eye(3))

    # Instantiate the network
    model = hk.without_apply_rng(hk.transform(model_fn))  
    
    rng_seq = hk.PRNGSequence(42)
    
    if FLAGS.diffusion_type == "vexp":
        noise_schedule = jnp.linspace(0.05, 1.5, 
                                      FLAGS.n_steps) 
        noise_schedule = noise_schedule**3 + 0.0001
    elif FLAGS.diffusion_type == "vpres":
        beta = jnp.linspace(0.0001, 0.08, FLAGS.n_steps)
        # This corresponds to alpha, it starts from 1, i.e. almost no change in the image
        noise_schedule = jnp.cumprod(1 - beta)
    else:
        raise NotImplementedError     
 
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
        optimizer = optax.chain(
            optax.adam(learning_rate=FLAGS.learning_rate),
            optax.scale_by_schedule(lr_schedule)
        )
        opt_state = optimizer.init(params)

        # Define the loss function
        def loss_fn(params, batch):
            s = batch['sn+1'].reshape([-1,1])
            
            mu, scale = model.apply(params, batch['yn+1'], s)
             
            @jax.vmap
            def fn(x, y, mu, scale):
                mu = SO3(mu) @ SO3(y) # Apply residual rotation
                dist = IsotropicGaussianSO3(mu.wxyz, scale, 
                                            force_small_scale=True)
                return dist.log_prob(x)

            loss = -fn(batch['yn'], batch['yn+1'], mu, scale)

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
            batch = get_batch(next(dset), next(rng_seq), noise_schedule)
            # Sampling another batch if the current one had a NaN
            while jnp.isnan(batch['yn+1']).any():
                print("WARNING: Skipped a batch because it had a NaN")
                batch = get_batch(next(dset), next(rng_seq), noise_schedule)
            
            loss, params, opt_state = update(params, opt_state, batch)
 
            if jnp.isnan(loss):
                print("Got a NaN during training!")
                break

            if step%50==0:
                summary_writer.scalar('train_loss', loss, step)
                summary_writer.scalar('learning_rate', FLAGS.learning_rate*lr_schedule(step), step)

            if step%10000 ==0:
                with open(output_dir+ '/' + FLAGS.dataset + '_model-%d.pckl'%step, 'wb') as file:
                    pickle.dump(params, file)

        summary_writer.flush()

        with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'wb') as file:
            pickle.dump(params, file)

    with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'rb') as file:
        params = pickle.load(file)

    # Starting sampling from the trained model
    if FLAGS.diffusion_type == "vexp":
        X0 = jax.vmap(lambda k: SO3.sample_uniform(k).wxyz)(jax.random.split(next(rng_seq), FLAGS.test_nsamples))
    elif FLAGS.diffusion_type == "vpres":
        X0 = IsotropicGaussianSO3([1,0,0,0], 1).sample(FLAGS.test_nsamples,seed=next(rng_seq))
    else:
        raise NotImplementedError

    @jax.jit
    @jax.vmap
    def fn_sample(x, delta_mu, s,key):
        return IsotropicGaussianSO3((SO3(delta_mu) @ SO3(x)).wxyz, s, force_small_scale=True).sample(seed=key)

    x_t = X0
    for sn in noise_schedule[::-1]:
        mu, s = model.apply(params, 
                            x_t,
                            sn*jnp.ones([FLAGS.test_nsamples,1]))

        x_t = fn_sample(x_t, mu, s, jax.random.split(next(rng_seq), FLAGS.test_nsamples))

    # Remove nans if we accidentally sampled any
    x_t = x_t[~onp.isnan(x_t.sum(axis=-1))]
    with open(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
        onp.save(f, x_t)

    visualize_so3_density(jax.vmap(lambda q: SO3(q).as_matrix())(x_t), 100);
    plt.savefig(output_dir + FLAGS.dataset + '_VExp_' + str(FLAGS.test_nsamples) + ".png")

    if FLAGS.compute_c2st:    
        true_samp_loc = 'reference_distribution/' + FLAGS.dataset + '_true_200_000.npy'

        with open(true_samp_loc , 'rb') as file:
            true_samp = onp.load(file)

        seed = 1
        if true_samp.shape[1] == 3:
            true_samp = jax.vmap(lambda m: SO3.from_matrix(m).wxyz )(true_samp) # print(X.shape)

        visualize_so3_density(jax.vmap(lambda q: SO3(q).as_matrix())(true_samp), 100);
        plt.savefig(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + "_true.png")
        print("Calculating c2st ... ")

        c2_score = c2st(true_samp, x_t, seed, FLAGS.n_folds)

        with open(output_dir+"output.txt", "a") as f:
            print( "C2ST score: "+ str(c2_score), file=f)

        
        print("\n")
        print("\n")


        print("C2ST score: "+ str(c2_score))


if __name__ == "__main__":
    app.run(main)
