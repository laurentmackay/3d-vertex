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

from Task2 import fmags, dt, tau

fmags = np.array(fmags)



red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]




patterns = [f'Task2_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle' for fmag in fmags]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

def square_length(G,t):
    return euclidean_distance(G.nodes[4]['pos'], G.nodes[3]['pos'])

def square_width(G,t):
    return euclidean_distance(G.nodes[4]['pos'], G.nodes[2]['pos'])

results_l1 = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_length,
                            indices=(-1,))

results_l3 = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_width,
                            indices=(-1,))

cols = 3


linewidth=2

l20=const.default_edge['l_rest']
l10=l20/np.sqrt(2)
l30=l10
k=const.mu_apical









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

plt.plot(phi, results_l1[:,-1]/l10-1, label='numerics', linewidth=linewidth)
plt.plot(phi, theo_l1-1,'--', label='theoretical',linewidth=linewidth)

plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\Delta_1}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)
plt.xlim((0,4))
plt.ylim((0,1.6))
plt.legend()
plt.tight_layout()
fig.savefig(f'Task2_validation_l1.png', dpi=200)

fig = plt.figure(2)
fig.set_size_inches(6, 4)

plt.plot(phi, -(results_l3[:,-1]/l10-1), label='numerics', linewidth=linewidth)
plt.plot(phi, -(theo_l3-1),'--',label='theoretical',linewidth=linewidth)

plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\Delta_2}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)
plt.xlim((0,6))
plt.ylim((0,0.55))

plt.legend()
plt.tight_layout()
fig.savefig(f'Task2_validation_l3.png', dpi=200)

plt.show()
