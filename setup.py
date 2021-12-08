import sys
import os
from setuptools import setup, find_packages



install_requires=['numpy>1.10','scipy','numba>0.51', 'vtk==9.0.3','matplotlib', 'mayavi']


setup(name='3d-vertex',
      version='1.0',
      packages=find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links
      )