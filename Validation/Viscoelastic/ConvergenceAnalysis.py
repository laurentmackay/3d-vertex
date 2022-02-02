import numpy as np
import matplotlib.pyplot as plt

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.funcs import euclidean_distance


if __name__ == '__main__':
    def square_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])


    patterns =('compression_*.pickle','compression_large_*.pickle', 'extension_*.pickle','extension_large_*.pickle')
    patterns=('compression_*.pickle','compression_large_*.pickle', 'periodic_*.pickle','extension_large_*.pickle')

    dts = (0.5, 0.1, 0.05, 0.01, 0.005, 0.001)
    patterns = [ f'extntions_dt_{dt}_*.pickle' for dt in dts]
    results = analyze_networks(path='./data/viscoelastic_convergence/',
                               patterns=patterns,
                               func=square_length)


    
    f=1
    linewidth=3
    max_error=[]
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(12, 8)
    axs = axs.flatten()
    for res, ax  in zip(results, axs):
        if res:
            res=np.array(res)
            t=res[:,0]
            ax.plot(t, res[:,1],linewidth=linewidth, label='numerical')
            # tau = 0.5*const.eta/const.mu_apical

            lam = 2*const.mu_apical/const.eta + 1/60
            gamma = const.mu_apical/const.eta * (2*f)
            B=(2.0/(lam*const.eta))*(0+gamma/lam)
            sol = (3.4+B)+gamma*(1/const.mu_apical-2.0/(const.eta*lam))*t - B*np.exp(-lam*t)
            ax.plot(t, sol,linewidth=linewidth, label='numerical')

            max_error.append(np.max(np.abs(sol-res[:,1])))
        else:
            max_error.append(np.nan)


    fig = plt.figure(2)
    fig.set_size_inches(12, 8)
    plt.loglog(dts, max_error, '--',linewidth=2, marker='o')
    
    plt.xlabel('dt', fontsize=14)
    plt.ylabel('max error', fontsize=14)
    #  fig.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # fig.title.set_fontsize(16)
    # fig.legend(loc='right')
    plt.grid(which='minor',linestyle='--', alpha=0.15, color='k')
    plt.grid(which='major',linestyle='--', color='k')
    plt.tight_layout()
    plt.show()