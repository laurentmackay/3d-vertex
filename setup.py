import sys
import os
from setuptools import setup, find_packages



install_requires=['networkx','numpy>1.10','scipy','numba>=0.54', 'vtk==9.0.3','matplotlib', 'mayavi','dill']
dependency_links = ['git+https://github.com/enthought/mayavi.git@master','git+https://github.com/uqfoundation/dill.git@master']

setup(name='3d-vertex',
      version='1.0',
      packages=find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links
      )