import string

import numpy as np
import matplotlib.pyplot as plt

from VertexTissue import globals as const
from VertexTissue.Analysis import *

from VertexTissue.funcs import euclidean_distance

from periodic_2D import f, omega


if __name__ == '__main__':

    square_length = lambda  G, t  : euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

    fig = plt.figure()
    fig.set_size_inches(6, 4)
   

    patterns=[]
    i=0

    ke=const.mu_apical
    kv = ke*60
    eta = const.eta
    kappa = eta + 2*kv



    linewidth=3






    arg = (kappa*ke/(omega*kv**2)+omega*eta/ke)/2
    delta = -np.arctan(arg)

    num = ke**2+(omega*kv)**2
    denom = (kappa*omega*ke)**2+(kv*eta*omega**2)**2


    denom2 = (kappa*ke)**3+kappa*ke*(omega*kv*eta)**2

    res = analyze_network_evolution(path='./data/viscoelastic/',
                            pattern=f'periodic_transient_force_{f}_freq_{omega}_*.pickle',
                            func=square_length)

    res=np.array(res)
    t=res[:,0]
    plt.plot(t, res[:,1],linewidth=linewidth, label='numerical')
    A=f
    lam = 2*ke/eta + 1/60
    gamma = ke/eta * (2*A)
    B=(2.0/(lam*eta))*(0+gamma/lam)
    sol = (3.4+B)+gamma*(1/const.mu_apical-2.0/(const.eta*lam))*t - B*np.exp(-lam*t)

    

    num2 = -kv*2*A*omega*ke*eta*kv**2
    l_final = const.l_apical + 2*A/(2*omega*kv+omega*eta)
    l_trans = -2*np.exp(-lam*t)*(num2)/denom2
    amp = 2*A*np.sqrt(num/denom)
    sol = l_final+amp*np.sin(omega*t+delta) +l_trans


    plt.plot(t, sol, '--',linewidth=linewidth, label='theoretical')
    
    plt.xlabel('t (seconds)', fontsize=14)
    plt.ylabel('length', fontsize=14)
    plt.xlim(0,150)
    # ax.title.set_text(f'({lbl}) Force={f}, max error = {np.max(np.abs(sol-res[:,1])):.3e}')
    # ax.title.set_fontsize(16)
    plt.legend()

    
    
    plt.tight_layout()
    plt.savefig('./Validation/Viscoelastic/viscoelastic_periodic_timeseries.png', dpi=300)
    plt.show()