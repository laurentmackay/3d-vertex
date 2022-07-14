from itertools import product
import sys

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool
import networkx as nx
from VertexTissue import funcs

from VertexTissue.funcs import euclidean_distance, get_pos_array, get_points, convex_hull_volume_bis, unit_vector
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from Task1 import fmags, dt, tau

fmags = np.array(fmags)

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def rectify_network_positions(G):
    pos_dict=nx.get_node_attributes(G,'pos')
    pos_dict = {k:v[0:2] for k,v in pos_dict.items()}

    arr=np.array([*pos_dict.values()])
    arr-=arr[0]
    theta = np.arccos(np.dot(unit_vector(arr[0],arr[3]),(1,0) ))

    R=rotation_matrix(np.pi/2-theta)

    return {k:np.matmul(R, v) for k,v in pos_dict.items()}

red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]




patterns = [f'Task1_{fmag}_{tau}_eta_{1}_dt_{dt}_*.pickle' for fmag in fmags]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[3]['pos'])

results = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_length,
                            indices=(-1,))

cols = 3

fig = plt.figure(1)
fig.set_size_inches(6, 4)

linewidth=3.0

l10=const.default_edge['l_rest']
l20=l10/np.sqrt(2)
k=const.mu_apical
fcrit=(l10-l20)/k







theo = np.zeros(fmags.shape)



inds = fmags<fcrit
theo[inds] = (l20+fmags[inds]/k)
theo[~inds] =((l20+2*l10)+fmags[~inds]/(k))/3

phi = fmags/(k*l20)

plt.plot(phi, results[:,-1]/(2*l20),label='numerics', linewidth=linewidth)
plt.plot(phi, theo/l20,'--',label='theoretical',linewidth=linewidth)


fs=16
plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\ell_1}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)

plt.tight_layout()
plt.legend()
fig.savefig('Task1_validation.png', dpi=200)
plt.show()
