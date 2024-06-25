from itertools import product
import sys

import networkx as nx
import numpy as np
from pathos.pools import ProcessPool as Pool
from VertexTissue.util import rectify_network_positions

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import hex_hex_grid, square_grid_2d, tissue_3d, T1_minimal
from VertexTissue.T1 import simple_T1
import VertexTissue.SG as SG
from VertexTissue.Geometry import unit_vector, unit_vector_2D
import VertexTissue.globals as const
from VertexTissue.PyQtViz import edge_view



from VertexTissue.globals import default_ab_linker, default_edge

N=1
M=1

tau=60

fmag=1

dt=.01

fmags = np.hstack((np.linspace(0, 2.2, 5),np.linspace(2.2, 2.8, 11)[1:],np.linspace(2.9, 5, 5))) * 3.4 
# fmags=np.linspace(0, 5, 20)* 3.4 / np.sqrt(2)

const.eta=10

def run(fmag):

    edge_attr = default_edge.copy()
    linker_attr = default_ab_linker.copy()
    spoke_attr = default_edge.copy()

    print(f'eta = {const.eta}')
    _ , G = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
    # pos = rectify_network_positions(G)
    # pos = {k:np.array([*v,0.0]) for k,v in pos.items()}
    # nx.set_node_attributes(G, pos, 'pos')
    
    # edge_view(G,exec=True, cell_edges_only=False)

    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = (11,1)

    keep_forcing=True
    def stop_forcing(*args):
        nonlocal keep_forcing
        keep_forcing=False

    def forcing(t,force_dict):

        pos_a = G.nodes[11]['pos']
        pos_b = G.nodes[1]['pos']

        force_vec = fmag*unit_vector(pos_a, pos_b)
        forces=[force_vec, -force_vec]

        if keep_forcing:
            for p,f in zip(forced_points, forces):
                force_dict[p] += f

        # if t<=600:
        #     force_dict[16][1] = 0.1
        #     force_dict[15][1] = 0.1

        #     force_dict[19][1] = -0.1
        #     force_dict[18][1] = -0.1

        # for i in (1,2,4):
        #     force_dict[i][1] = 0 
            # 



    #create integrator
    integrate = monolayer_integrator(G, G, post_callback=forcing, intercalation_callback=stop_forcing, ndim=3, viewer={'cell_edges_only':False,'draw_axes':True}, save_rate=-1, maxwell=False, minimal=True)
    t_final=500

    pattern=f'data/elastic/Task3_T1_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
    # pattern=None

    integrate(dt, t_final, save_pattern=pattern)
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done f={fmag}')


if __name__ == '__main__':

    run(1.5*3.4)
    # Pool(nodes=6).map( run, fmags)
    