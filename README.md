# 3d-vertex

## About
This repository contains code that can be used in a 3D Vertex Model simulation of tissue mechanics (see an example [here](#pickle-player-demo)). It is fork of the code developed by:
* [Clinton H. Durney](https://clintondurney.github.io/)

The python package `VertexTissue` extends and improves that code in the following ways:
+ It allows for viscoelastic disipation in the cell-edges:
	* using either Maxwell Model or Standard Linear Solid Model elements
+ Significant improvements to performance (~100-fold faster):
	* Jit-compilation of core numerical routines (using `numba`)
	* Simplification of some core loops and optimzation of datatypes
+ Cell-volumes are computed using an exact formula rather than the quickhull algorithm (requires the use of the `fastvol=True` during integration).
+ Adaptive timestepping that detects large force differences and keeps deformations below user-specified thresholds. 

The code contained herein was written for the simulations in the following publication:

*Stress-relaxation in a self-constricting model of tubulogenesis during Drosophila salivary gland invagination*


## Dependencies

Plotting the tissue geometry uses OpenGL and requires the `OpenGL Utility Toolkit`. This can be obtain from the `freeglut` package,  on debian-based linux distributions this can usually be installed using:
```
sudo apt install freeglut*
```

## Installation ### 

Clone this repo locally and run :
```
pip install .
```

You may also add the `-e` flag to install as an editable project  or alternatively use any other installation tool that is compatible with a `pyproject.toml` specification file.

## Example

Run `main.py` in the root folder of this repository.

```
pip main.py
```

## Package Overview

A more in-depth description of the package and its features/options can be found [here](OVERVIEW.md).
