import numpy as np
import matplotlib.pyplot as plt
from Validation.Viscoelastic.TimeseriesAnalysis import get_theo_length

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.Geometry import euclidean_distance, get_pos

from shear_rod import dts, tau

if __name__ == '__main__':
    def square_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])



    patterns = [ f'shear_rod_{tau}_dt_{dt}_*.pickle' for dt in (0.01, )]
    results = analyze_networks(path='./data/viscoelastic/',
                               patterns=patterns,
                               func= lambda G,t: get_pos(G))


    
 
    linewidth=3
    max_error=[]

    sol = np.array([[*e[-1].values()] for e in results[-1]])
    t = np.array([e[0] for e in results[-1]])

    fig = plt.figure(1)
    fig.set_size_inches(6, 4)


    plt.plot(t,sol[:,0,1], color='k', linewidth=1 )

    
    fig = plt.figure(2)
    fig.set_size_inches(6, 4)


    plt.plot(sol[:,0,0],sol[:,0,1], color='k', linewidth=1 )

    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    #  fig.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # fig.title.set_fontsize(16)
    # fig.legend(loc='right')
    # plt.grid(which='minor',linestyle='--', alpha=0.15, color='k')
    # plt.grid(which='major',linestyle='--', color='k')
    
    
    plt.tight_layout()
    # plt.savefig('./Validation/Viscoelastic/viscoelastic_convergence.png', dpi=300)
    plt.show()