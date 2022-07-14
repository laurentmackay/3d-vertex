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

from constant_force import fmags


patterns = [f'rod_force_{fmag}_*.pickle' for fmag in fmags]
# patterns=[]
# for eta in (1, 10, 100, 1000):
#     patterns.extend([f'diagon_{i}_{tau}_eta_{eta}_dt_{dt}_*.pickle' for i in (0,2,7,12)])

def length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[1]['pos'])

def tension(G,t):
    return G[0][1]['tension']

results = analyze_networks(path='./data/tension/',
                            patterns=patterns,
                            func=length,
                            indices=(-1,))



plt.plot(fmags,results[:,1],color='k',label='Tension Remodelling Element')
plt.plot(fmags,1-fmags,linestyle=':',color='k',label='Elastic Element')
plt.legend(fontsize=12)
plt.xlabel('Nondimensional Force', fontsize=16)
plt.ylabel('Final Length',fontsize=16)
plt.savefig('tension_remodelling.png', dpi=200)
plt.show()