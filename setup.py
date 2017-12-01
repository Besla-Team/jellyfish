#! /usr/bin/env python

from setuptools import setup

setup(name='Jellyfish',
      version='0.1',
      description='Tools to plot and analyze N-body simulations of hosts and satellites galaxies',
      author='Ekta Patel and Nicolas Garavito',
      author_email='jngaravitoc@email.arizona.edu',
      install_requieres=['numpy', 'scipy', 'matplotlib', 'astropy', 'pygadgetreader'],
      packages=['jellyfish'],
)
