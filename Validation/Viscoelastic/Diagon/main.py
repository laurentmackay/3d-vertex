from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector, unit_vector_2D
import VertexTissue.globals as const
from VertexTissue.PyQtViz import edge_view




N=1
M=1

tau=60

fmag=1

dt=10

visco_edges = [((1,3),),
               ((0,1),),
               ((0,1),(1,3)),
               ((0,1),(1,4)),
               ((0,1),(1,3),(1,4)),
               ((0,1),(1,3),(1,4),(0,4)),
               ((0,1),(1,3),(1,4),(0,4),(4,3)),
               ((0,1),(1,3),(0,4),(4,3)),
               ((0,1),(0,2)),
               ((0,1),(0,2),(1,4)),
               ((0,1),(0,2),(1,4),(2,4)),
               ((0,4),(3,4)),
               ((0,1),(2,3)),
               ((0,1),(1,3),(3,2),(2,0)),
               ((0,4),(4,3),(2,4),(1,4))
               ]


const.eta=1000
def run(i):
    
    G = square_grid_2d( 1, 1, spokes=True)
    # edge_view(G, exec=True)
    for a,b in visco_edges[i]:
        G[a][b]['tau'] = tau


    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ 0 , 3]
    pos_a = G.nodes[0]['pos']
    pos_b = G.nodes[3]['pos']

    force_vec = fmag*unit_vector(pos_a, pos_b)

    # forced_points=[range(N+1)]

    forces=[-force_vec, force_vec]

    def forcing(t,force_dict):
        
        for p,f in zip(forced_points, forces):
            force_dict[p] += f

        force_dict[0][:] = 0
            # 



    #create integrator
    integrate = monolayer_integrator(G, G, post_callback=forcing, ndim=3, viewer=False, save_rate=100, maxwell=True, minimal=True)
    t_final=100000
    integrate(dt,t_final, adaptive=True, save_pattern=f'data/viscoelastic/diagon_{i}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle')
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done i={i}')


if __name__ == '__main__':

    # run(-1)
    Pool(nodes=6).map( run, range(len(visco_edges)))
    