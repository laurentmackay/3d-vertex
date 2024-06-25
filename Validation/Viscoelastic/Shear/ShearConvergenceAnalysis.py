import numpy as np
import matplotlib.pyplot as plt
from Validation.Viscoelastic.TimeseriesAnalysis import get_theo_length

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.Geometry import euclidean_distance, get_pos

from shear_2 import dts, tau

if __name__ == '__main__':
    def square_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])



    patterns = [ f'shear_2_{tau}_dt_{dt}_*.pickle' for dt in dts]
    results = analyze_networks(path='./data/viscoelastic/',
                               patterns=patterns,
                               func= lambda G,t: get_pos(G))


    
 
    linewidth=3
    max_error=[]
    fig, axs = plt.subplots(2,4)
    fig.set_size_inches(12, 8)
    axs = axs.flatten()

    sol = np.array([[*e[-1].values()] for e in results[-1]])
    t = np.array([e[0] for e in results[-1]])
    for res, ax  in zip(results, axs):
        if res:

            res = np.array([[*e[-1].values()] for e in res])
            err = np.sqrt(np.sum(np.power(res[:sol.shape[0],:,:] - sol, 2),axis=(2,1)))
            ax.plot(t, err,linewidth=linewidth, label='error')
            # tau = 0.5*const.eta/const.mu_apical


            max_error.append(np.max(err))
            # max_error.append(np.max(np.abs(sol[-1]-res[-1,1])))
        else:
            max_error.append(np.nan)

    
    fig = plt.figure(2)
    fig.set_size_inches(6, 4)

    x=np.logspace(-.8,-4.2)
    plt.loglog(x,x *max_error[0]/dts[0], color='k', linewidth=1 )

    plt.loglog(dts, max_error, '--',linewidth=2, marker='o')
    
    plt.xlabel('dt', fontsize=14)
    plt.ylabel('max error', fontsize=14)
    #  fig.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # fig.title.set_fontsize(16)
    # fig.legend(loc='right')
    plt.grid(which='minor',linestyle='--', alpha=0.15, color='k')
    plt.grid(which='major',linestyle='--', color='k')
    
    
    plt.tight_layout()
    # plt.savefig('./Validation/Viscoelastic/viscoelastic_convergence.png', dpi=300)
    plt.show()