from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import vertex_integrator
from VertexTissue.Tissue import square_grid_2d
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector_2D
from VertexTissue.globals import default_ab_linker, default_edge
from VertexTissue.PyQtViz import edge_view

from shear_2 import dts


N=2
M=2

tau=60

fmag=1

dt=0.1

def run(dt):

    G = square_grid_2d( N, M)
    # edge_view(G, exec=True)
    for a,b in ((1,4),(4,7)):
        G[a][b]['tau'] = tau


    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ *l1 , *l2]
    pos_a = G.nodes[0]['pos']
    pos_b = G.nodes[1]['pos']

    force_vec = fmag*unit_vector_2D(pos_a, pos_b)

    # forced_points=[range(N+1)]

    forces=[*[ -force_vec for _ in l1], *[force_vec for _ in l2]]

    def forcing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f
            # force_dict[p][0] = 0



    #create integrator
    integrate = vertex_integrator(G, G, post_callback=forcing, ndim=2, viewer=True, save_rate=1, maxwell=True)
    t_final=8000
    integrate(dt, t_final, save_pattern=f'data/viscoelastic/shear_1_{tau}_dt_{dt}_*.pickle')
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done dt={dt}')


if __name__ == '__main__':

    run(0.1)
    # Pool(nodes=6).map( run, dts)
    