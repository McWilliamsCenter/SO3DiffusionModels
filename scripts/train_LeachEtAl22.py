import pickle
import sys
import os
sys.path.append('diffusion-extensions/')

# Main script used to train a particular model on a particular dataset.
from absl import app
from absl import flags

import numpy as np
import torch
import torch.nn as nn

from diffusion import SO3Diffusion
from models import SinusoidalPosEmb

from util import *
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard
from so3dm.plotting import visualize_so3_density
import matplotlib.pyplot as plt

flags.DEFINE_string("dataset", "checkerboard", "Dataset to train on. Can be 'checkerboard'.")
flags.DEFINE_string("output_dir", "models/leachetal", "Folder where to store model and training info.")
flags.DEFINE_integer("batch_size", 512, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 400000, "Total number of training steps.")
flags.DEFINE_bool("train", True, "Whether to train the model or just sample from trained model.")
flags.DEFINE_integer("test_nsamples", 20000, "Number of samples to draw at testing time.")

FLAGS = flags.FLAGS

class RotPredict(nn.Module):
    def __init__(self, d_model=255, out_type="rotmat", in_type = "rotmat"):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        if self.in_type == "rotmat":
            in_channels = 9
            t_emb_dim  = d_model - in_channels
        if self.out_type == "skewvec":
            self.d_out = 3
        elif self.out_type == "rotmat":
            self.d_out = 6
        else:
            RuntimeError(f"Unexpected out_type: {out_type}")

        self.time_embedding = SinusoidalPosEmb(t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.d_out),
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x_flat = torch.flatten(x, start_dim=-2)
        t_emb = self.time_embedding(t)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x_flat.shape[0], -1)
        xt = torch.cat((x_flat,t_emb), dim=-1)

        out = self.net(xt)
        if self.out_type == "rotmat":
            out = six2rmat(out)
        return out

def main(_):
    output_dir = FLAGS.output_dir+"_"+FLAGS.dataset

    torch.set_anomaly_enabled(True)
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = RotPredict(out_type="skewvec").to(device)

    if FLAGS.train:
        net.train()
        process = SO3Diffusion(net, loss_type="skewvec").to(device)

        optim = torch.optim.Adam(process.denoise_fn.parameters(), lr=3e-4)

        # Just to make sure pytorch is initialized before TF
        torch.matrix_exp(torch.randn(3,3))

        # Load the dataset
        dset = tfds.load(FLAGS.dataset,split='train')
        dset = dset.repeat()
        dset = dset.shuffle(buffer_size=10000)
        dset = dset.batch(FLAGS.batch_size)
        dset = dset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dset = dset.as_numpy_iterator()

        summary_writer = tensorboard.SummaryWriter(output_dir)

        for step in tqdm(range(FLAGS.training_steps)):
            truepos = torch.from_numpy(next(dset)['pos_mat'])
            truepos = truepos.to(device)
            loss = process(truepos)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 50 == 0:
                summary_writer.scalar('train_loss', loss.item(), step)
            if step % 1000 == 0:
                torch.save(net.state_dict(), output_dir+"/weights_so3.pt")
    
    net.load_state_dict(torch.load(output_dir+"/weights_so3.pt", map_location=device))
    net.eval()
    process = SO3Diffusion(net, loss_type="skewvec").to(device)

    with torch.no_grad():
        # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
        R, _ = torch.qr(torch.randn((FLAGS.test_nsamples, 3, 3)))
        
        for i in tqdm(reversed(range(0, process.num_timesteps)),
                    desc='sampling loop time step',
                    total=process.num_timesteps,
                    ):
            R = process.p_sample(R.to(device), torch.full((1,), i, device=device, dtype=torch.long))

    R = R.to('cpu').numpy()
    with open(output_dir+"/Rsamples.npy", "wb") as f:
        np.save(f, R)
    visualize_so3_density(R,32);
    plt.savefig(output_dir+"/Rsamples.png")

if __name__ == "__main__":
    app.run(main)