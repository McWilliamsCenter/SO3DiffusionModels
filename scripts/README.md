# Training and Testing Scripts


## Installation

In order to be able to run all the comparitive experiments, you should install 
the external projects with the following:
```bash
# For denoising diffusion models
pip install git+https://github.com/lucidrains/denoising-diffusion-pytorch.git
pip install git+https://github.com/lucidrains/se3-transformer-pytorch.git
```

This assumes you already have pytorch installed.

## Running training

Running training for the different models
```bash
python train_LeachEtAl22.py
python train_so3dm.py
```
This will create folders under 'models' with samples of the models after training.