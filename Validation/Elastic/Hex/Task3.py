from itertools import product
import sys

import networkx as nx
import numpy as np
from pathos.pools import ProcessPool as Pool
from VertexTissue.util import rectify_network_positions

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import hex_hex_grid, square_grid_2d, tissue_3d
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector, unit_vector_2D
import VertexTissue.globals as const
from VertexTissue.PyQtViz import edge_view




N=1
M=1

tau=60

fmag=1

dt=.01
switch1=1.5
switch2=2.5

fmags = np.hstack((np.linspace(0, switch1, 5),np.linspace(switch1, switch2, 26)[1:],np.linspace(switch2, 5, 6)[1:])) * 3.4 
# fmags=np.linspace(0, 5, 20)* 3.4 / np.sqrt(2)

ndim=2

const.eta=10
def run(fmag):
    print(f'eta = {const.eta}')
    G = square_grid_2d( 1, 1, spokes=True)
    G, _ = tissue_3d(basal=False, hex=1)
    # pos = rectify_network_positions(G)
    # pos = {k:np.array([*v,0.0]) for k,v in pos.items()}
    # nx.set_node_attributes(G, pos, 'pos')
    
    # edge_view(G,exec=True, cell_edges_only=False)

    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ 4 , 1]
    pos_a = G.nodes[4]['pos']
    pos_b = G.nodes[1]['pos']

    force_vec = fmag*unit_vector(pos_a, pos_b)[:ndim]

    # forced_points=[range(N+1)]

    forces=[-force_vec, force_vec]

    e43 = np.array([0.,1.,0.])

    def forcing(t,force_dict):
        
        for p,f in zip(forced_points, forces):
            force_dict[p] += f

        
        force_dict[4][:] = 0

        # for i in (1,2,4):
        #     force_dict[i][1] = 0 
            # 



    #create integrator
    viewer_dict = {'cell_edges_only':False}
    integrate = monolayer_integrator(G, G, post_callback=forcing, ndim=ndim, viewer=viewer_dict, save_rate=-1, maxwell=False, minimal=True)
    t_final=20000

    pattern=f'data/elastic/Task3_hex_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
    # pattern=None

    integrate(dt, t_final, save_pattern=pattern)
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done f={fmag}')


if __name__ == '__main__':

    run(1.85*3.4)
    # Pool(nodes=6).map( run, fmags)
    