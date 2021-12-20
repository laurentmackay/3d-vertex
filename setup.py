import sys
from platform import uname

from setuptools import setup, find_packages

IS_LINUX = sys.platform.startswith('linux')
IS_WINDOWS = sys.platform.startswith('win')

def in_wsl() -> bool:
    return 'microsoft-standard' in uname().release


install_requires=['networkx','numpy>1.10','scipy','numba>=0.54','pyqtgraph','pyopengl']
dependency_links =[]


# if not in_wsl():
#       install_requires.append('mayavi')
#       dependency_links.append('git+https://github.com/enthought/mayavi.git@master')

if IS_WINDOWS:
      install_requires.append('dill')
      dependency_links.append('git+https://github.com/uqfoundation/dill.git@master')

setup(name='3d-vertex',
      version='1.0',
      packages=find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links
      )