from itertools import product
import sys

import networkx as nx
import numpy as np
from pathos.pools import ProcessPool as Pool
from VertexTissue.util import rectify_network_positions

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d
import VertexTissue.SG as SG
from VertexTissue.Geometry import unit_vector, unit_vector_2D
import VertexTissue.globals as const
from VertexTissue.PyQtViz import edge_view




N=1
M=1

tau=60

fmag=1

dt=.1
ndim=2

fmags = np.hstack((np.linspace(0, 2.2, 10),np.linspace(2.28, 2.55, 31)[1:],np.linspace(2.8, 5, 10))) * 3.4 / np.sqrt(2)
# fmags = np.linspace(2.28, 2.55, 31) * 3.4 / np.sqrt(2)
# fmags=np.linspace(0, 5, 20)* 3.4 / np.sqrt(2)

const.eta=75
def run(fmag):
    print(f'eta = {const.eta}')
    G = square_grid_2d( 1, 1, spokes=True)
    pos = rectify_network_positions(G)
    pos = {k:np.array([*v,0.0]) for k,v in pos.items()}
    nx.set_node_attributes(G, pos, 'pos')
    
    # edge_view(G,exec=True)

    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ 0 , 3]
    pos_a = G.nodes[0]['pos']
    pos_b = G.nodes[3]['pos']

    force_vec = fmag*unit_vector(pos_a, pos_b)[:ndim]

    # forced_points=[range(N+1)]

    forces=[-force_vec, force_vec]

    e43 = np.array([0.,1.,0.])

    def forcing(t,force_dict):
        
        for p,f in zip(forced_points, forces):
            force_dict[p] += f

        
        force_dict[0][:] = 0

        # for i in (1,2,4):
        #     force_dict[i][1] = 0 
            # 



    #create integrator
    integrate = monolayer_integrator(G, G, post_callback=forcing, ndim=ndim, viewer=True, save_rate=1, maxwell=False, minimal=True)
    t_final=800000
    pattern=f'data/elastic/Task3_vid_asymmetric_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
    # pattern=None

    integrate(dt, t_final, adaptive=True, save_pattern=pattern)
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done f={fmag}')


if __name__ == '__main__':

    run(2.45*3.4/np.sqrt(2))
    # Pool(nodes=6).map( run, fmags)
    