import numpy as np
import matplotlib.pyplot as plt
from Validation.Elastic.TimeseriesAnalysis import get_theo_length

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.Geometry import euclidean_distance

from constant_force_2D import dts, fmag

if __name__ == '__main__':
    def square_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])



    patterns = [ f'extension_{fmag}_dt_{dt}_*.pickle' for dt in dts]
    results = analyze_networks(path='./data/elastic/',
                               patterns=patterns,
                               func=square_length)


    
    f=fmag/2
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

            sol = get_theo_length(t, fmag)
            ax.plot(t, sol,linewidth=linewidth, label='numerical')

            max_error.append(np.max(np.abs(sol-res[:,1])))
        else:
            max_error.append(np.nan)

    
    fig = plt.figure(2)
    fig.set_size_inches(6, 4)

    x=np.logspace(-.8,-4.2)
    plt.loglog(x,x *max_error[-1]/dts[-1], color='k', linewidth=1 )

    plt.loglog(dts, max_error, '--',linewidth=2, marker='o')
    
    plt.xlabel('dt', fontsize=14)
    plt.ylabel('max error', fontsize=14)
    #  fig.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # fig.title.set_fontsize(16)
    # fig.legend(loc='right')
    plt.grid(which='minor',linestyle='--', alpha=0.15, color='k')
    plt.grid(which='major',linestyle='--', color='k')
    
    
    plt.tight_layout()
    plt.savefig('./Validation/Elastic/elastic_convergence.png', dpi=300)
    plt.show()