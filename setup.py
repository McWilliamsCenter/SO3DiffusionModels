from setuptools import setup, find_packages

setup(
    name='SO3DM',
    version='0.0.1',
    url='https://github.com/McWilliamsCenter/SO3DiffusionModels',
    author='Yesukhei Jagvaral',
    description='Collection of tools for implementing SO3 diffusion models',
    packages=find_packages(),
    install_requires=['optax', 'dm-haiku', 'healpy',
                      'tensorflow-probability', 
                      'tensorflow-datasets'
                      'jaxlie', 'flax'],
)
