from itertools import product
import sys
from matplotlib import markers
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool
import networkx as nx
from VertexTissue import funcs

from VertexTissue.funcs import euclidean_distance, get_pos_array, get_points, convex_hull_volume_bis, unit_vector
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from Task3 import fmags, dt, tau

fmags = np.array(fmags)



red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]




patterns = [f'Task3_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle' for fmag in fmags]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[3]['pos'])/2

def square_width(G,t):
    return euclidean_distance(G.nodes[1]['pos'], G.nodes[2]['pos'])/2
a = np.array([1.0,0.,0.])
def square_epsilon(G,t):
    a = unit_vector(G.nodes[1]['pos'], G.nodes[2]['pos'])
    b = unit_vector(G.nodes[1]['pos'], G.nodes[4]['pos'])
    return np.arccos(np.sum(a*b))

results_epsilon = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_epsilon,
                            indices=(-1,))

results_l3 = analyze_networks(path='./data/elastic/',
                            patterns=patterns,
                            func=square_width,
                            indices=(-1,))

cols = 3


# axs=axs.flatten()

linewidth=2

l20=const.default_edge['l_rest']
l10=l20/np.sqrt(2)
l30=l10
k=const.mu_apical




theo = np.zeros(fmags.shape)

l10=const.default_edge['l_rest']/np.sqrt(2)
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
trans=2.31
theo_l1=(2*gamma2/np.sqrt(1+omega**2) + (1 + phi)  )/3
theo_l3=(2*gamma2*omega/np.sqrt(1+omega**2) + gamma3  )/3

fig = plt.figure(1)
fig.set_size_inches(6, 4)

plt.plot(phi, 2*(results_l3[:,-1]/l10), label='numerics', linewidth=linewidth)
plt.plot(phi, 2*(theo_l3),'--',label='Task 2 Prediction',linewidth=linewidth)


plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'Network Height / $\ell_0$', rotation='vertical', fontsize=fs, labelpad=12)


plt.tight_layout()
plt.legend()
# axs[1].legend()
fig.savefig(f'Task3_validation_height.png', dpi=200)


fig = plt.figure(2)
fig.set_size_inches(6, 4)

# plt.plot((trans,trans),(0,0.5))

plt.plot(phi, results_epsilon[:,-1]/np.pi, label='numerics', linewidth=linewidth)
ax1=plt.gca()
plt.legend()
plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\epsilon}{\pi}$', rotation='horizontal', fontsize=fs, labelpad=12)


ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1

ip = InsetPosition(ax1, [0.6,0.1,0.35,0.8])

ax2.set_axes_locator(ip)
# mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec='0.5')

inds=np.logical_and(phi>2.25 , phi<2.54)
ax2.plot(phi[inds],results_epsilon[inds,-1]/np.pi)




plt.tight_layout()

# axs[1].legend()
fig.savefig(f'Task3_validation_epsilon.png', dpi=200)

fig = plt.figure(3)
fig.set_size_inches(6, 4)

force = fmags[32]

pattern  = f'Task3_{force}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
res = analyze_networks(path='./data/elastic/',
                            patterns=(pattern,),
                            indices=(-1,))

print(f'phi = {force/(3.4/np.sqrt(2))}')

G=res[0][-1][-1]
pos=rectify_network_positions(G, phi=-np.pi/2)
pos = {k:v[:2] for k,v in pos.items()}

nx.drawing.nx_pylab.draw_networkx_edges(G, pos)

fig.savefig(f'Task3_partially_buckled.png', dpi=200)
plt.tight_layout()
plt.show()
