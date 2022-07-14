from itertools import product
import sys

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.funcs import euclidean_distance
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from constant_force_2D import dts, fmag

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

def get_theo_length(t, fmag):
    tau = 0.5*const.eta/const.mu_apical
    return 3.4-(fmag)*(np.exp(-t/tau)-1)

def get_length(dt):
    pattern = f'/extension_dt_{dt}_*.pickle'

    if fmag>=0:
        pattern = f'extension_{fmag}_dt_{dt}_*.pickle'
    else:
        pattern = f'compression_{fmag}_dt_{dt}_*.pickle'

    results = analyze_network_evolution(path='./data/elastic/',
                               pattern=pattern,
                               func=square_length)
    return results

if __name__ == '__main__':

    # patterns=('compression_*.pickle','compression_large_*.pickle', 'expansion_*.pickle','extension_large_*.pickle')
    # patterns = ('compression_*.pickle','compression_large_*.pickle', 'extension_periodic_*.pickle','extension_large_periodic_*.pickle')


    res = get_length(dts[0])

    fig = plt.figure()
    fig.set_size_inches(6, 4)
    # axs = axs.flatten()
    # force=[-1,-2.5, 1, 2.5]
    linewidth=3
    # for res, ax, f, lbl in zip(results, axs, force, ('a','b', 'c','d')):
    res=np.array(res)
    t=res[:,0]
    plt.plot(t, res[:,1],linewidth=linewidth, label='numerical')

    sol = get_theo_length(t, fmag)
    # sol = 3.4-(fmag)*(np.cos(t/5))/10

    plt.plot(t, sol, '--',linewidth=linewidth, label='theoretical')
    plt.xlim(0,600)
    plt.xlabel('t (seconds)', fontsize=14)
    plt.ylabel('length', fontsize=14)
    # ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # ax.title.set_fontsize(16)
    plt.legend(loc='right')

    
    plt.tight_layout()
    plt.savefig('./Validation/Elastic/elastic_timeseries.png', dpi=300)
    plt.show(block=True)