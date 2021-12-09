import sys
import os
from setuptools import setup, find_packages



install_requires=['networkx','numpy>1.10','scipy','numba>=0.54', 'vtk==9.0.3','matplotlib', 'mayavi']


setup(name='3d-vertex',
      version='1.0',
      packages=find_packages(),
      install_requires=install_requires
      )