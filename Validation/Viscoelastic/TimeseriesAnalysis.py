import string

import numpy as np
import matplotlib.pyplot as plt

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.funcs import euclidean_distance

from constant_2D import forces


if __name__ == '__main__':

    square_length = lambda  G, t  : euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

    fig, axs = plt.subplots(2,int(np.ceil(len(forces)/2)))
    fig.set_size_inches(12, 8)
    axs = axs.flatten()

    patterns=[]
    i=0

    ke=const.mu_apical
    kv = ke*60
    eta = const.eta
    kappa = eta + 2*kv

    alphabet_list = list(string.ascii_lowercase)


    linewidth=3
    for f in forces:

        ax=axs[i]
        lbl = alphabet_list[i]
        i+=1



        res = analyze_network_evolution(path='./data/viscoelastic/',
                                pattern=f'force_{f}_*.pickle',
                                func=square_length)

        res=np.array(res)
        t=res[:,0]
        ax.plot(t, res[:,1],linewidth=linewidth, label='numerical')
        A=f
        lam = 2*ke/eta + 1/60
        gamma = ke/eta * (2*A)
        B=(2.0/(lam*eta))*(0+gamma/lam)
        sol = (3.4+B)+gamma*(1/const.mu_apical-2.0/(const.eta*lam))*t - B*np.exp(-lam*t)

        



        ax.plot(t, sol, '--',linewidth=linewidth, label='theoretical')
        
        ax.set_xlabel('t (seconds)', fontsize=14)
        ax.set_ylabel('length', fontsize=14)
        ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
        ax.title.set_fontsize(16)
        ax.legend(loc='right')

    
    
   






    
    plt.tight_layout()
    plt.show()