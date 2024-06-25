from itertools import product
import sys

import matplotlib.pyplot as plt

import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.Geometry import euclidean_distance, get_pos, get_points, convex_hull_volume
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from constant_force_2D import dts, fmag

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

def cell_volumes(G,t, inds=None):
    centers = G.graph['centers']
    if inds:
        centers=centers[inds]
    pos = get_pos(G)
    return tuple(convex_hull_volume(get_points(G, c, pos)) for c in centers)




def get_volumes(dt):
    pattern = f'3D_intercalation_{dt}_*.pickle'


    results = analyze_network_evolution(path='./data/T1/',
                               pattern=pattern,
                               func=cell_volumes)
    return results

if __name__ == '__main__':

    # patterns=('compression_*.pickle','compression_large_*.pickle', 'expansion_*.pickle','extension_large_*.pickle')
    # patterns = ('compression_*.pickle','compression_large_*.pickle', 'extension_periodic_*.pickle','extension_large_periodic_*.pickle')


    res = get_volumes(dts[0])

    fig = plt.figure()
    fig.set_size_inches(6, 4)
    # axs = axs.flatten()
    # force=[-1,-2.5, 1, 2.5]
    linewidth=3
    # for res, ax, f, lbl in zip(results, axs, force, ('a','b', 'c','d')):
    t=np.array([r[0] for r in res])
    res=np.array([r[1] for r in res])
    
    plt.plot(t, res[:,:], linewidth=linewidth, label='numerical')

    sol = get_theo_length(t, fmag)
    # sol = 3.4-(fmag)*(np.cos(t/5))/10

    # plt.plot(t, sol, '--', linewidth=linewidth, label='theoretical')
    plt.xlim(0,600)
    plt.xlabel('t (seconds)', fontsize=14)
    plt.ylabel('Volume (um^3)', fontsize=14)
    # ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # ax.title.set_fontsize(16)
    # plt.legend(loc='right')

    
    plt.tight_layout()
    # plt.savefig('./Validation/Elastic/elastic_timeseries.png', dpi=300)
    plt.show(block=True)