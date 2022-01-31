import numpy as np
import matplotlib.pyplot as plt
import globals as const
from Analysis import *



if __name__ == '__main__':
    def square_length(G,t):
        return euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

    results = analyze_networks(path='./data/viscoelastic/',
                               patterns=('compression_*.pickle','compression_large_*.pickle', 'extension_*.pickle','extension_large_*.pickle'),
                               func=square_length)

    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12, 8)
    axs = axs.flatten()
    force=[-1,-2.5, 1, 2.5]
    linewidth=3
    for res, ax, f, lbl in zip(results, axs, force, ('a','b', 'c','d')):
        if res:
            res=np.array(res)
            t=res[:,0]
            ax.plot(t, res[:,1],linewidth=linewidth, label='numerical')
            tau = 0.5*const.eta/const.mu_apical

            lam = 2*const.mu_apical/const.eta + 1/60
            gamma = const.mu_apical/const.eta * (2*f)
            B=(2.0/(lam*const.eta))*(0+gamma/lam)
            sol = (3.4+B)+gamma*(1/const.mu_apical-2.0/(const.eta*lam))*t - B*np.exp(-lam*t)
            ax.plot(t, sol, '--',linewidth=linewidth, label='theoretical')
            
            ax.set_xlabel('t (seconds)', fontsize=14)
            ax.set_ylabel('length', fontsize=14)
            ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
            ax.title.set_fontsize(16)
            ax.legend(loc='right')

    
    plt.tight_layout()
    plt.show()