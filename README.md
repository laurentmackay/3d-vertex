# 3d-vertex

## About
This repository contains code that can be used in a 3D Vertex Model simulation of tissue mechanics. It is heavily based on code developed by:
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

Stress-relaxation in a self-constricting model of tubulogenesis during *Drosophila* salivary gland invagination


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



### Basic Tissue Data Structure

Tissues/Cells are represented in the center-and-spoke cell-vertex model pioneered by the Feng research group. This is implemented as a`networkx` graph data-structure with cell center vertices connected to its outer vertices by spokes 


* see `VertexTissue.Tissue.tissue_3d`
	* creates a monolayer tissue from a hexagonal tilings of cells

__Critical Graph Properties__

The simulations require graphs with a few custom properties in order to work. These properties define the cells in the tissue as well as the mechanical properties of the cell-cell interfaces between them (i.e., edges connecting outer vertices).

 The cells/tissue are defined by three main properties in a graph `G` : 
* `G.graph['centers']` stores the center vertex of each cell (`np.array`)
* `G.graph['circum_sorted']` stores the corresponding list of outer vertices
* `G.graph['triangles']` array of triangle indices for surface bending calculations
 
These properties are automatically populated by `VertexTissue.Tissue.tissue_3d` and are updated after changes in network topology using `VertexTissue.Tissue.new_topology`.
	 
 The edge between vertices `i` and `j` is `e=G[i][j]` and it has:
* `e['myosin']` which linearly controls the contractile strength
* `e['tau']` controls the edge's relaxation timescale (viscoelastic only).

These edge properties can be updated dynamically at any point in the simulation. The "standard" way of doing this in a time-dependent manner is by passing a `ResearchTools.Events.TimeBasedEventExecutor` as a callback function to the integrator (see keywords `pre_callback` and `post_callback`  in `VertexTissue.vertex_3d.monolayer_integrator`).
 
 

### Time Integration

Tissue geometries are evolved in time using a `VertexTissue.vertex_3d.monolayer_integrator` which takes a graph `G` as a required parameter as well as many possible optional keywords (see the docstring for details). Once constructed, an integrator can then be called with two required parameters:
* `dt` a (maximum) timestep,
* `t_final` a final integration time.

 Most integration keywords can be specified either during integrator construction or at the actual integration call. Exceptions to this rule are any keywords that control tissue mechanics, such as `ndim`, `minimal`, `SLS`, `maxwell`, `maxwell_nonlin`, and `fastvol`.

__Adaptive timestepping__ may decrease the timestep in order to simultaneously satisfy three tolerance criteria:
* `length_rel_tol` a tolerance for the change in length relative to the current length,
* `length_abs_tol` a tolerance for the absolute change in length,
* `angle_tol` an angular tolerance.

 If all three tolerances are satisfied, the timestep is iteratively increased by a fraction `adaptation_rate` at each timestep until it reaches its maximum `dt`. A lower bound on the timestep is specified by `dt_min`.



### Integration Results

Integration will return a `dict()` with `float` keys that map to snapshots of the evolving graph geometry at approximately fixed time-intervals. The time-interval between snapshots is specified by the `save_rate` keyword. 

Snapshots can also be saved to storage as pickle files using the `save_pattern` keyword to specify the save path. If `save_patten` contains a \* symbol then each snapshot is saved as an individual pickle file with the \*  replaced by the current time, otherwise a single dict() containing all snapshots is saved to the path.



### Intercalations

If the `T1` keyword is set to `True`, then at each timestep we check all edge lengths to see if they are below a threshold (specified by the parameter `VertexTissue.globals.l_intercalation`). If any edges are below the threshold, we perform a topological transition in the network of cell-edges to allow for cell intercalation. Any edge that should not be allowed to undergo such topological transitions can be specified in the `blacklist` keyword.

### Cell-Edge Mechanics

By default all cell-edges are treated as elastic rods, this can be modified by using either the `maxwell` or `SLS` keywords to use either Maxwell or Standard Linear Solid elements instead (both are `False` by default).
* The viscoelastic relaxation timescale of each edge is then controlled by the `tau` parameter of each edge (default value is set at launch by the `VertextTissue.globals.tau` parameter). 
* If `SLS` is non-`False` its value should be between `0.0` and `1.0` to specify the stiffness of the element's Maxwell branch (as a complement, the Hookean branch has stiffness `1.0-SLS`).
* For both types of viscoelastic elements, non-linear rest-length adaptation can be specified by passing a function to the `maxwell_nonlin` keyword. 
	* This function  must take three `numpy.array`parameters `ell`, `L`, `L0` which specify the current length of each edge, its current rest-legnth, and a reference rest-length, respectively.
	*  It should return the quantity $`\tau \frac{{\rm d}L}{{\rm d}t}`$ for all the rest-lengths $`L`$ in the graph as a single `numpy.array`.

### Visualization


Interactive visualization of network geometry is done using the `VertextTissue.PyQTViz.edge_view` and `VertextTissue.PyQTViz.edge_viewer` functions which perform subtly different roles. The `edge_view` function requires a graph `G`, and will display its geometry using a customized `pyqtgraph.opengl.GLGraphicsItem` widget as well as provide controls to modify which edges are displayed and their appearance. The `edge_viewer` does the same, but additionally returns function that can be passed more graphs objects to dynamically update the visualization. 

If the `viewer` keyword of a `monolayer_itegrator`is not `False`, an `edge_viewer` will be spawned to visualize the evolution of the tissue geometry in real-time. A `dict()` of keywords values can also be passed to `viewer` to customize the default appearance of the spawned `edge_viewer`.

A series of snapshot can be viewed as "movie" by using the `VertexTissue.Player.pickle_player` function. You may specify a filepath for pickle files with the `pattern` keyword, which uses the same syntax as the `monolayer_integrator` keyword `save_pattern` (i.e., individual pickle files for each graph or one large pickle file containing a `dict()` of graphs). Alternatively, you may directly pass a `dict()` returned as the result of `monolayer_integrator` by using the `save_dict` keyword.





