from itertools import product
import sys
from matplotlib.markers import MarkerStyle

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool
import networkx as nx
from VertexTissue import funcs

from VertexTissue.funcs import euclidean_distance, get_pos_array, get_points, convex_hull_volume_bis, unit_vector
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from Task4 import fmags, dt, taus


fmags = np.array(fmags)



red=(1.0,0.0,0.0)
black=(0.0,0.0,0.0)
def edge_coloring(G):
    return [ red if np.isfinite(G[e[0]][e[1]]['tau']) else black for e in G.edges()]

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[3]['pos'])/2

def square_width(G,t):
    return euclidean_distance(G.nodes[1]['pos'], G.nodes[2]['pos'])/2

a = np.array([1.0,0.,0.])
def square_epsilon(G,t):
    # a = unit_vector(G.nodes[1]['pos'], G.nodes[2]['pos'])
    b = unit_vector(G.nodes[1]['pos'], G.nodes[4]['pos'])
    return np.arccos(np.sum(a*b))



fig1 = plt.figure(1)
fig1.set_size_inches(6, 4)

fig2 = plt.figure(2)
fig2.set_size_inches(6, 4)

fig3 = plt.figure(3)
fig3.set_size_inches(6, 4)

for tau in taus:

    patterns = [f'Task4_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle' for fmag in fmags]
    # patterns=[]
    # for eta in (1, 10, 100, 1000):
    #     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

    # results_l1 = analyze_networks(path='./data/elastic/',
    #                             patterns=patterns,
    #                             func=square_length,
    #                             indices=(-1,))

    results_l3 = analyze_networks(path='./data/elastic/',
                                patterns=patterns,
                                func=square_width,
                                indices=(-1,))


    # results_epsilon = analyze_networks(path='./data/elastic/',
    #                             patterns=patterns,
    #                             func=square_epsilon,
    #                             indices=(-1,))




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

    theo_l1=(2*gamma2/np.sqrt(1+omega**2) + (1 + phi)  )/3
    theo_l3=(2*gamma2*omega/np.sqrt(1+omega**2) + gamma3  )/3

    # plt.figure(1)

    # plt.plot(phi, results_l1[:,-1]/l10-1, label=r'numerics ($\tau$='+str(tau)+')', linewidth=linewidth)


    plt.figure(3)

    plt.plot(phi, results_l3[:,-1]/l10, label=r'numerics ($\tau$='+str(tau)+')', linewidth=linewidth)






    # plt.figure(2)

    # plt.plot(phi, results_epsilon[:,-1]/np.pi, label=r'numerics ($\tau$='+str(tau)+')', linewidth=linewidth)







# axs[1].plot(phi, (results_l3[:,-1]/l10), label='numerics', linewidth=linewidth)
# axs[1].plot(phi, (theo_l3),'--',label='theoretical',linewidth=linewidth)
# axs[1].plot((2.29099,2.29099), (0,1),'--',label='predicted instability',linewidth=linewidth)

# axs[1].set_xlabel('$\phi$', fontsize=fs)
# axs[1].set_ylabel(r'$\frac{\Delta_2}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)
# # axs[1].set_xlim((0,6))
# # axs[1].set_ylim((0,0.55))

plt.figure(1)
plt.plot(phi, theo_l1-1,'--',label='Task 2 Prediction',linewidth=linewidth)
plt.plot(phi, theo/l10-1,'--',label='Task 1 Prediction',linewidth=linewidth)

plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\Delta_1}{\ell_0}$', rotation='horizontal', fontsize=fs, labelpad=12)

plt.tight_layout()
plt.legend()
fig1.savefig(f'Task5_validation_delta1.png', dpi=200)


plt.figure(2)
plt.xlabel('$\phi$', fontsize=fs)
plt.ylabel(r'$\frac{\epsilon}{\pi}$', rotation='horizontal', fontsize=fs, labelpad=12)
plt.ylim((-.01,0.5))

plt.tight_layout()
plt.legend()
fig2.savefig(f'Task5_validation_epsilon.png', dpi=200)


plt.figure(3)
plt.plot(phi, theo_l3,'--',label='Unbuckled Prediction',linewidth=linewidth)


plt.tight_layout()
plt.legend()
# plt.plot(phi, theo/l10-1,'--',label='Task 1 Prediction',linewidth=linewidth)

plt.show()
