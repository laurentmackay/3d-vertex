from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector_2D
from VertexTissue.globals import default_ab_linker, default_edge
import VertexTissue.globals as const

N=1
M=1
dts = np.logspace(-1,-4, 7)
# dts = np.logspace(-1,-1, 1)
fmag=1

const.eta = 10

def run(dt):

    G = square_grid_2d( N, M)
    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ *l1 , *l2]
    pos_a = G.nodes[forced_points[0]]['pos']
    pos_b = G.nodes[forced_points[-(M+1)]]['pos']

    force_vec = fmag*unit_vector_2D(pos_a, pos_b)

    for a,b in zip(l1,l2):
        G[a][b]['tau'] = 60

    # forced_points=[range(N+1)]

    forces=[-force_vec,  -force_vec,   force_vec,  force_vec]

    def forcing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f



    #create integrator
    integrate = monolayer_integrator(G, G, pre_callback=forcing, ndim=2, player=False, save_rate=1, maxwell=True)
    t_final=100
    #integrate
    if fmag>=0:
        integrate(dt, t_final, save_pattern=f'data/viscoelastic/extension_{fmag}_dt_{dt}_*.pickle')
    else:
        integrate(dt, t_final, save_pattern=f'data/viscoelastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done dt={dt}')


if __name__ == '__main__':

    # initialize the tissue
    
    pool = Pool(nodes=6)

    

    pool.map( run, dts)
    