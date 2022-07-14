from itertools import product
import sys

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool
import networkx as nx

from VertexTissue.funcs import euclidean_distance, get_pos_array, get_points, convex_hull_volume_bis, unit_vector
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from main import visco_edges, dt, tau



red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]




patterns = [f'diagon_{i}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle' for i in range(len(visco_edges))]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

results = analyze_networks(path='./data/viscoelastic/',
                            patterns=patterns,
                            indices=(0,-1))

cols = 3
rows = int(np.ceil(len(results)/cols))
fig, axs = plt.subplots(rows, cols)
fig.set_size_inches(12, 8)
axs = axs.flatten()

i=0
for res, ax in zip(results, axs):
    if res is not None:
        G0=res[0][-1]
        pos0 = rectify_network_positions(G0)

        G1=res[1][-1]
        t1=res[1][0]
        pos1 = rectify_network_positions(G1)
        bbx0=np.max(np.array([*pos0.values()])[:,0])
        bbx1=np.abs(np.min(np.array([*pos1.values()])[:,0]))

        space=1.2

        for k,v in pos1.items():
            pos1[k][0]+=bbx0+bbx1+space
        
        # np.mean(pos1arr)

        # np.mean(pos1arr)

        G = nx.disjoint_union(G0,G1)


        pos1 = {k+len(G0):v for k,v in pos1.items()}

        pos = {**pos0, **pos1}
        ec=edge_coloring(G)
        nx.drawing.nx_pylab.draw_networkx_edges(G, pos, ax = ax, edge_color=ec)
        ax.set_title(f'{i} (t={int(np.round(t1))})')
        i+=1
    # nx.drawing.nx_pylab.draw_networkx_edges(G1, pos1, ax = ax)

plt.tight_layout()
fig.savefig('pulled_squares_visco.png', dpi=100)
plt.show()
