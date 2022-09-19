from itertools import product
import sys

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool
import networkx as nx
from VertexTissue import Geometry

from VertexTissue.Geometry import euclidean_distance, get_pos_array, get_points, convex_hull_volume, unit_vector
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from Task3 import fmags, dt, tau

fmags = np.array(fmags)



red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]




patterns = [f'Task3_hex_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle' for fmag in fmags]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

def square_length(G,t):
    return euclidean_distance(G.nodes[1]['pos'], G.nodes[4]['pos'])/2

def square_height(G,t):
    return euclidean_distance(G.nodes[3]['pos'], G.nodes[5]['pos'])/2

def hex_angle(G,t):
    a = unit_vector(G.nodes[0]['pos'], G.nodes[1]['pos'])
    b = unit_vector(G.nodes[0]['pos'], G.nodes[2]['pos'])
    return np.arccos(np.sum(a*b))

results_l1 = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_length,
                            indices=(-1,))

results_l3 = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_height,
                            indices=(-1,))

results_epsilon = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=hex_angle,
                            indices=(-1,))


# axs=axs.flatten()

linewidth=2

l20=const.default_edge['l_rest']
l10=l20/np.sqrt(2)
l30=l10
k=const.mu_apical




theo = np.zeros(fmags.shape)

l10=const.default_edge['l_rest']
l20=const.default_edge['l_rest']
k=const.mu_apical
fcrit=(l20-l10)/k


inds = fmags<fcrit
theo[inds] = (l10+fmags[inds]/k)
theo[~inds] =((l10+2*l20)+fmags[~inds]/(k))/3





theo_l1 = np.zeros(fmags.shape)
theo_l3 = np.zeros(fmags.shape)

f0=fmags


phi = f0/(k*l10)

gamma3=l30/l10
gamma2=l20/l10

omega = gamma3/(1+phi)

fs=16

theo_l1=(2*gamma2/np.sqrt(1+omega**2) + (1 + phi)  )/3
theo_l3=(2*gamma2*omega/np.sqrt(1+omega**2) + gamma3  )/3
fig = plt.figure(1)
fig.set_size_inches(6, 4)

plt.plot(phi, results_l3[:,-1]/l10, label='numerics', linewidth=linewidth)


plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\Delta_1}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)

plt.tight_layout()
plt.legend()
fig.savefig(f'Task3_hex_validation_height.png', dpi=200)

fig = plt.figure(2)
fig.set_size_inches(6, 4)

plt.plot(phi, results_epsilon[:,-1]/np.pi, label='numerics', linewidth=linewidth)


plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\epsilon}{\pi}$', rotation='horizontal', fontsize=fs, labelpad=12)
# axs[0].set_xlim((0,3))
plt.ylim((-.01,0.5))

fig.savefig(f'Task3_hex_validation_angle.png', dpi=200)

plt.tight_layout()
plt.legend()
# fig.savefig(f'Task3_validation_epsilon.png', dpi=200)
plt.show()