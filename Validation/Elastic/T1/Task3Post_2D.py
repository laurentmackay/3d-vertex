from itertools import product
import pickle
import sys

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathos.pools import ProcessPool as Pool
from VertexTissue.util import rectify_network_positions

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import hex_hex_grid, square_grid_2d, tissue_3d, T1_minimal
from VertexTissue.T1 import simple_T1
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector, unit_vector_2D
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
const.l_intercalation=0.0
def run(fmag):

    edge_attr = default_edge.copy()
    linker_attr = default_ab_linker.copy()
    spoke_attr = default_edge.copy()

    
    with open('./data/elastic/Post_T1.pickle', 'rb') as file:
            G = pickle.load( file)

    fig = plt.figure(1)
    pos_dict=nx.get_node_attributes(G,'pos')
    pos_dict = {k:v[0:2] for k,v in pos_dict.items()}

    
    centers=(0,7,14,17)
    grey=(0.9,0.9,0.9)
    black=(0.0,0.0,0.0)
    def edge_coloring(G):
        return [ grey if e[1] in centers or e[0] in centers   else black for e in G.edges()]
    ec=edge_coloring(G)


    nx.drawing.nx_pylab.draw_networkx_edges(G, pos_dict, ax=plt.gca(), edge_color=ec)

    plt.tight_layout()
    fig.savefig('PostT1.pdf', dpi=100)
    plt.show()

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
    root2=np.sqrt(2)
    compressed = (16,15,9,3,19,18,13,5)
    origin = np.array((0.,0.,0.))
    def forcing(t,force_dict):

        pos_a = G.nodes[11]['pos']
        pos_b = G.nodes[1]['pos']

        force_vec = 2.8*unit_vector(pos_a, pos_b)
        forces=[force_vec, -force_vec]

        if t<200 and keep_forcing:
            for p,f in zip(forced_points, forces):
                force_dict[p] += f

        if t<=1600:
            for i in compressed:
                b = G.nodes[i]['pos']
                v = unit_vector(b, origin)
                force_dict[i]+= fmag*v



        # for i in (1,2,4):
        #     force_dict[i][1] = 0 
            # 



    #create integrator
    integrate = monolayer_integrator(G, G, post_callback=forcing, intercalation_callback=stop_forcing, ndim=3, viewer={'cell_edges_only':False,'draw_axes':True}, save_rate=-1, maxwell=False, minimal=True)
    t_final=5000

    pattern=f'data/elastic/Task3_T1_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
    pattern=None

    integrate(dt, t_final, save_pattern=pattern)
    #integrate
    # if fmag>=0:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/shear_{fmag}_dt_{dt}_*.pickle')
    # else:
    #     integrate(dt, t_final, save_pattern=f'data/elastic/compression_{fmag}_dt_{dt}_*.pickle')
    print(f'Done f={fmag}')


if __name__ == '__main__':

    run(.8)
    # Pool(nodes=6).map( run, fmags)
    