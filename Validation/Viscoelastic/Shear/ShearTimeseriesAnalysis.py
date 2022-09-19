import numpy as np
import matplotlib.pyplot as plt
from Validation.Viscoelastic.TimeseriesAnalysis import get_theo_length

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.Geometry import euclidean_distance, get_pos, unit_vector


dt = 0.1
tau = 60

if __name__ == '__main__':
    def square_diagonal_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[8]['pos'])

    def square_angle(G,t):
        a = G.nodes[0]['pos']
        b = G.nodes[1]['pos']
        c = G.nodes[2]['pos']
        A = unit_vector(b,a)
        B = unit_vector(b,c)

        return np.arccos(np.sum(A*B))

    def geometry_summary(G,t):
        return (square_angle(G,t), square_diagonal_length(G,t))


    lbls = (1,2,3,4,5,6,7,8,9,'all')
    patterns = [ f'shear_{l}_{tau}_dt_{dt}_*.pickle' for l in lbls]
    results = analyze_networks(path='./data/viscoelastic/',
                               patterns = patterns,
                               func = geometry_summary)


    
 



    linewidth=1
    max_error=[]


    # sol = np.array([[*e[-1]] for e in results[-1]])
    # t = np.array([e[0] for e in results[-1]])

    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(12, 8)
    axs = axs.flatten()
    i=0
    for  res, lbl in zip(results, lbls):
        if res is not None:
            # res =np.array(res)
            # t = np.array([e[0] for e in res])
            # res = np.array([[*e[-1]] for e in res])
            # err = np.sqrt(np.sum(np.power(res[:sol.shape[0],:,:] - sol, 2),axis=(2,1)))
            axs[0].plot(res[:,0], res[:,1]+0.00001*i, linewidth=linewidth, alpha=0.7, label=lbl)
            axs[1].plot(res[:,0], res[:,2]+0.00001*i, linewidth=linewidth, alpha=0.7, label=lbl)
            i+=1
        #     # tau = 0.5*const.eta/const.mu_apical


        #     # max_error.append(np.max(err))
        #     # max_error.append(np.max(np.abs(sol[-1]-res[-1,1])))
        # else:
        #     max_error.append(np.nan)

    
    # fig = plt.figure(2)
    # fig.set_size_inches(6, 4)

    # x=np.logspace(-.8,-4.2)
    # plt.loglog(x,x *max_error[0]/dts[0], color='k', linewidth=1 )

    # plt.loglog(dts, max_error, '--',linewidth=2, marker='o')
    
    # plt.xlabel('dt', fontsize=14)
    # plt.ylabel('max error', fontsize=14)
    # #  fig.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # # fig.title.set_fontsize(16)
    axs[0].legend(loc='right')
    axs[1].legend(loc='right')
    # plt.grid(which='minor',linestyle='--', alpha=0.15, color='k')
    # plt.grid(which='major',linestyle='--', color='k')
    
    
    plt.tight_layout()
    # plt.savefig('./Validation/Viscoelastic/viscoelastic_convergence.png', dpi=300)
    plt.show()