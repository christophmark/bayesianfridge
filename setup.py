#!/usr/bin/env python

from setuptools import setup

setup(
    name='bayesianfridge',
    packages=['bayesianfridge'],
    version='1.0',
    description='Sequential Monte Carlo sampler for PyMC2 models.',
    url='https://github.com/christophmark/bayesianfridge',
    download_url = 'https://github.com/christophmark/bayesianfridge/tarball/1.0',
    author='Christoph Mark',
    author_email='christoph.mark@fau.de',
    license='The MIT License (MIT)',
    install_requires=['numpy>=1.12.0', 'scipy>=0.19.0', 'pymc>=2.3.6', 'tqdm>=4.10.0'],
    keywords = ['SMC', 'MCMC', 'Monte Carlo', 'Metropolis', 'annealing', 'marginal likelihood', 'model evidence'],
    classifiers = [],
    )
