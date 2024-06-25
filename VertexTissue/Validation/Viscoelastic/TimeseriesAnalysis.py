from itertools import product
import sys

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.Geometry import euclidean_distance
import VertexTissue.globals as const
from VertexTissue.Analysis import *
from VertexTissue.util import *

from constant_force_2D import dts, fmag

ke=const.mu_apical
kv = ke*60
eta = const.eta
kappa = eta + 2*kv

def square_length(G,t):
    return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

def get_theo_length(t, fmag):
    A=fmag
    lam = 2*ke/eta + 1/60
    gamma = ke/eta * (2*A)
    B=(2.0/(lam*eta))*(0+gamma/lam)
    sol = (3.4+B)+gamma*(1/const.mu_apical-2.0/(const.eta*lam))*t - B*np.exp(-lam*t)
    return sol

def get_length(dt):
    
    if fmag>=0:
        pattern = f'extension_{fmag}_dt_{dt}_*.pickle'
    else:
        pattern = f'compression_{fmag}_dt_{dt}_*.pickle'

    results = analyze_network_evolution(path='./data/viscoelastic/',
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
    
    res=np.array(res)
    t=res[:,0]
    plt.plot(t, res[:,1],linewidth=linewidth, label='numerical')

    sol = get_theo_length(t, fmag)
    # sol = 3.4-(fmag)*(np.cos(t/5))/10

    plt.plot(t, sol, '--',linewidth=linewidth, label='theoretical')
    plt.xlim(0,100)
    plt.xlabel('t (seconds)', fontsize=14)
    plt.ylabel('length', fontsize=14)
    # ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # ax.title.set_fontsize(16)
    plt.legend(loc='right')

    
    plt.tight_layout()
    plt.savefig('./Validation/Viscoelastic/viscoelastic_timeseries.png', dpi=300)
    plt.show(block=True)