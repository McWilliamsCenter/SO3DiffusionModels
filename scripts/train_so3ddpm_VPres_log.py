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

import pickle

from so3dm.metrics import c2st

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/so3ddpm_VPres_log/", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 1024, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 400_000 , "Total number of training steps.")
flags.DEFINE_bool("train", False, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 200_000, "Number of samples to draw at testing time.")

flags.DEFINE_bool("compute_c2st", True , "Whether to compute the c2st score agianst the true samples")

 
    
flags.DEFINE_integer("n_steps", 300, "Number of steps in noise schedule")
flags.DEFINE_float("start_noise", 0.05, "The value in which to start the noise schedule")
flags.DEFINE_float("stop_noise", 1.0, "The value in which to stop the noise schedule")
 
flags.DEFINE_integer("n_folds", 5, "Number of folds in c2st")

 
    
FLAGS = flags.FLAGS


 
def quat_power(quat, a):
    quat = SO3(quat)

    return SO3.exp(quat.log()*a).wxyz

def igso_sample(mu, sigma, key1 ):
    dist = IsotropicGaussianSO3( mu   , sigma)
    qn = dist.sample(seed=key1)
    return qn

@jax.jit
def get_batch(batch, key, noise_schedule, deltas  ):
    key1, key2, key3 = jax.random.split(key,3)
    # Sample from the target distribution
 
    @jax.vmap
    def sample(quaternion, temperature, seed):
        key1, key2 = jax.random.split(seed)
        def get_diffused_sample(quaternion,temperature, seed):
            beta =  noise_schedule[temperature] 
            oneMinus_beta =  1- beta 
            oneMinus_beta = jnp.clip(oneMinus_beta, 0)
        
            signal_scaled = jnp.where(jnp.sqrt(oneMinus_beta) < 0.5,
                                      quat_power( quaternion,  jnp.sqrt(oneMinus_beta)),
                                      quaternion )

            diffused_sample =  igso_sample( signal_scaled, beta, seed)
            return diffused_sample 


        diffused_sample =  get_diffused_sample(quaternion, temperature, key1)
        diffused_sample_plus1 =  get_diffused_sample(quaternion, temperature+1, key2)

        return {'x': quaternion, 'yn': diffused_sample, 'yn+1': diffused_sample_plus1, 
                'sn':noise_schedule[temperature ], 'sn+1':noise_schedule[temperature+1]}

    # Sample random noise levels from the schedule
    temp_list =  jnp.arange(len(noise_schedule))
    temperature = jax.random.choice(key3, temp_list, shape=[FLAGS.batch_size])
 

 
    # Sample random rotations
    return sample(batch['pos_quat'], temperature,  jax.random.split(key2, FLAGS.batch_size))

 
def model_fn(x,s):
    net = jnp.concatenate([x,s],axis=-1)
    net = hk.nets.MLP([256, 256, 256, 256, 256], activation=jax.nn.leaky_relu)(net)
    # We use a residual connection, because we expect deviations to be small
    mu = hk.Linear(3)(net) + x
 
    scale = jax.nn.softplus(hk.Linear(1)(net)) + 0.0001
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
    noise_schedule = noise_schedule**2 +0.001
    deltas = jnp.diff(noise_schedule, prepend=0)
     
 
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
#         with open(output_dir+'stripes3_model-10000.pckl' , 'rb') as file:
#             params = pickle.load(file)
        params = model.init(next(rng_seq), jnp.zeros([1,3]),jnp.zeros([1,1]))
         
        # Creating the optimizer
        optimizer = optax.chain(optax.adam(learning_rate=FLAGS.learning_rate))
        opt_state = optimizer.init(params)

        # Define the loss function
        def loss_fn(params, rng_key, batch):
            s = batch['sn+1'].reshape([-1,1])
            
            tangent_batch = SO3(batch['yn+1'])
            mu, scale = model.apply(params, jax.vmap(lambda x: SO3.log(x) )( tangent_batch), s)
             

            @jax.vmap
            def fn(x, mu, scale):
                mu = SO3.exp(mu)
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
            batch = get_batch(next(dset), next(rng_seq), noise_schedule, deltas)
            
            loss, params, opt_state = update(params, next(rng_seq), opt_state, batch)
 
            if jnp.isnan(loss):
#                 with open(output_dir+ '/' + FLAGS.dataset + '_model-%d.pckl'%(step%10000), 'wb') as file:
#                     params = pickle.load(file)
         
                break
                sys.exit()
            if step%50==0:
                summary_writer.scalar('train_loss', loss, step)
                #summary_writer.scalar('learning_rate', FLAGS.learning_rate*lr_schedule(step), step)

            if step%10000 ==0:
                with open(output_dir+ '/' + FLAGS.dataset + '_model-%d.pckl'%step, 'wb') as file:
                    pickle.dump(params, file)

        summary_writer.flush()

        with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'wb') as file:
            pickle.dump(params, file)
    
    #stripes3_model-250000.pckl
    #with open(output_dir+'/' + FLAGS.dataset + '_model-final.pckl', 'rb') as file:
    for i in [160000]:#[10000, 20000, 30000, 40000, 50000 , 60000, 70000, 80000, 90000, 100000 , 110000, 120000, 130000, 140000, 150000, 160000]:#, 170000, 180000, 190000]:
        with open(output_dir+'/stripes3_model-' +str(i) + '.pckl', 'rb') as file:
            params = pickle.load(file)




        # Starting sampling from the trained model
        X0 = IsotropicGaussianSO3([1,0,0,0], 1).sample(FLAGS.test_nsamples,seed=next(rng_seq)) #Normally distributed around the identity quaternion with sigma =1
        @jax.jit
        @jax.vmap
        def fn_sample(mu,s,key):
            mu = SO3.exp(mu)
            return IsotropicGaussianSO3(mu, s, force_small_scale=True).sample(seed=key)

        x_t = X0
        for variance in noise_schedule[::-1]:

            tangent_batch = SO3(x_t)

            mu, s = model.apply(params, 
                                jax.vmap(lambda x: SO3.log(x) )( tangent_batch),
                                jnp.sqrt(variance)*jnp.ones([FLAGS.test_nsamples,1]))


            x_t = fn_sample(mu, s, jax.random.split(next(rng_seq), FLAGS.test_nsamples))


        with open(output_dir + FLAGS.dataset + '_' + str(FLAGS.test_nsamples) + ".npy", "wb") as f:
            onp.save(f, x_t)

        visualize_so3_density(jax.vmap(lambda q: SO3(q).as_matrix())(x_t), 100);
        plt.savefig(output_dir + FLAGS.dataset + '_VPres_' + str(FLAGS.test_nsamples+i) + ".png")
    
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

        print(true_samp.shape[1])
        print("\n")
        print("\n")
        print("\n")
        print("\n")
        print("\n")

        print("C2ST score: "+ str(c2_score))


if __name__ == "__main__":
    app.run(main)