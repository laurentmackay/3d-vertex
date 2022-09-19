from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d, linear_grid
import VertexTissue.SG as SG
from VertexTissue.Geometry import unit_vector_2D
from VertexTissue.globals import default_ab_linker, default_edge
from VertexTissue.PyQtViz import edge_view

from shear_2 import dts


N=1
M=1

tau=60

fmag=1

dt=0.1

def run(dt):

    G = linear_grid( N)
    # edge_view(G, exec=True)
    for a,b in ((1,0), ):
        G[a][b]['tau'] = tau


    l1 = [0,]
    l2 = [1,]


    force_vec = fmag*np.array([0.0, 1.0])

    # forced_points=[range(N+1)]


    def forcing(t,force_dict):

        force_dict[0] += force_vec
        force_dict[1] -= force_vec
        # force_dict[1][:] = 0




    #create integrator
    integrate = monolayer_integrator(G, G, post_callback=forcing, ndim=2, viewer=False, save_rate=1, maxwell=True)
    t_final=1000
    integrate(dt, t_final, save_pattern=f'data/viscoelastic/shear_rod_{tau}_dt_{dt}_*.pickle')
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done dt={dt}')


if __name__ == '__main__':

    run(.01)
    # Pool(nodes=6).map( run, dts)
    