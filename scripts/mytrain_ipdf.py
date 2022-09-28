from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import pickle
import sys
import os
sys.path.append('google-research/')

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg
from flax.metrics import tensorboard
import matplotlib.pyplot as plt

tfkl = tf.keras.layers
eps = 1e-9

import os
from absl import app
from absl import flags
from absl import logging
import numpy as onp
import jax.numpy as jnp
from tqdm import tqdm
import tensorflow as tf
from implicit_pdf import data
from implicit_pdf import evaluation
from implicit_pdf import models

tfkl = tf.keras.layers
import tensorflow_datasets as tfds

from so3dm.plotting import visualize_so3_density

FLAGS = flags.FLAGS

#################################### I/O #######################################
flags.DEFINE_string('output_dir',
                    'models/ipdf',
                    'The directory in which to save results and images.')
flags.DEFINE_bool('save_models', True, 'Whether to save the vision and IPDF'
                  ' models at the end of training.')
flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
################################ Model Specs ###################################
flags.DEFINE_multi_integer('head_network_specs',
                           [256]*2,
                           'The sizes of the dense layers in the head network.')
flags.DEFINE_float('thresh', 0.001,
                     'This is the threshhold value to exclude the low probability points.')
#################################### Data ######################################
flags.DEFINE_multi_string('symsol_shapes', ['tet'],
                          'Can be any subset of the 8 shapes of SYMSOL I & II: '
                          'tet, cube, icosa, cyl, cone, tetX, cylO, sphereX, or'
                          ' \'symsol1\' for the first five.')
flags.DEFINE_integer('downsample_continuous_gt', 0,
                     'Whether, and how much, to downsample the cone and '
                     'cylinder ground truth rotations, which can make '
                     'evaluation slow.')
################################# Training #####################################
flags.DEFINE_integer('number_training_iterations',
                     100000,
                     'The number of iterations to train.')
flags.DEFINE_integer('number_eval_iterations', None,
                     'The number of iterations to eval.')

flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate.')
flags.DEFINE_integer('batch_size', 32, 'The batch size.')
flags.DEFINE_integer('test_batch_size', 32, 'The batch size for evaluation, '
                     'where it may be helpful to evaluate with a larger grid '
                     'and reduce this batch size if memory issues arise.')
flags.DEFINE_string('optimizer', 'Adam', 'The name of the optimizer to use.')
flags.DEFINE_integer('number_train_queries', 2**12,
                     'The number of sampled points on SO(3) for each loss '
                     'evaluation.')
flags.DEFINE_integer('number_eval_queries', 2**16,
                     'The number of sampled points on SO(3) to use for '
                     'evaluation.')
flags.DEFINE_enum('so3_sampling_mode', 'random',
                  ['random', 'grid'],
                  'How to sample from SO(3): \'random\' samples rotations '
                  'uniformly and \'grid\' creates an equivolumetric grid based '
                  'off Yershova et al. (2010).')
flags.DEFINE_integer('number_fourier_components', 1,
                     'The number of components in the positional encoding '
                     'for the implicit model.')
flags.DEFINE_integer('eval_every', -1, 'How often to evaluate.  If -1, evaluate'
                     '  100 times during training.')
flags.DEFINE_bool('skip_spread_evaluation', False, 'Whether to skip the '
                  'evaluation of the spread metric, which can be slow for '
                  'shapes with many ground truths.')
################################################################################
################################################################################

flags.DEFINE_bool('mock', False,
                  'Skip download of dataset and pre-trained weights. '
                  'Useful for testing.'
                  )


def main(_):
    output_dir = FLAGS.output_dir+"_"+FLAGS.dataset
    jnp.linalg.inv(jnp.eye(3))
    model_head = models.ImplicitSO3(1, FLAGS.number_fourier_components, [256,256,256,256],
                                  'random',
                                  FLAGS.number_train_queries,
                                  FLAGS.number_eval_queries)


    ######################   Load the datasets   ###############################
    batch_size = 32

    #dset = tfds.load('checkerboard', split="train")
    dset = tfds.load(FLAGS.dataset,split='train')
    dset = dset.repeat()
    dset = dset.shuffle(buffer_size=10000)
    dset = dset.batch(batch_size)
    dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dset = dset.as_numpy_iterator()
    _ = next(dset)

    summary_writer = tensorboard.SummaryWriter(output_dir)
    
    ##########################  Optimizer  #####################################
    optimizer = tf.keras.optimizers.get('Adam')
    learning_rate = 1e-5
    optimizer.learning_rate = learning_rate
    #########################  Logging setup  ##################################
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    log_dir = os.path.join('output_dir/', 'logs')
    train_summary_writer = tf.summary.create_file_writer(log_dir)


    def cosine_decay(step, warmup_steps=1000):
        warmup_factor = min(step, warmup_steps) / warmup_steps
        decay_step = max(step - warmup_steps, 0) / (
            number_training_iterations - warmup_steps)
        return learning_rate * warmup_factor * (1 + tf.cos(decay_step * np.pi)) / 2

    @tf.function
    def train_step( model_head, optimizer, rotations_gt):
        with tf.GradientTape() as tape:
            loss = model_head.compute_loss(tf.ones(shape=[batch_size, 1]), rotations_gt)

        grads = tape.gradient(loss, model_head.trainable_variables)
        optimizer.apply_gradients( zip(grads, model_head.trainable_variables))
        return loss

    number_training_iterations = FLAGS.number_training_iterations
    
    for step in tqdm(range( number_training_iterations)):
        step_num = optimizer.iterations.numpy()
        if step_num > number_training_iterations:
          break

        tf.keras.backend.set_value(optimizer.learning_rate,
                                   cosine_decay(step_num))

        #batch = get_batch(next(dset), next(rng_seq))
        rotations_gt = next(dset)['pos_mat']

        loss = train_step( model_head, optimizer, rotations_gt)

        train_loss(loss)

        if (step_num) % 100 == 0:
          avg_loss = train_loss.result()
          train_loss.reset_states()
          with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=step_num)
            tf.summary.scalar(
                'learning_rate', optimizer.learning_rate, step=step_num)
          logging.info('Step %d, training loss=%.2f', step_num, avg_loss)

    R, probs = model_head.output_pdf( num_queries=200_00 )
    R = R[probs>FLAGS.thresh]
    with open(output_dir+"/Rsamples.npy", "wb") as f:
        onp.save(f, R)
    visualize_so3_density(R,32);
    plt.savefig(output_dir+"/Rsamples.png")
    
    
if __name__ == '__main__':
  app.run(main)
