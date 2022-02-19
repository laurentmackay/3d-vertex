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

from visco_phase_shift import omegas, f

dev_null = open('/dev/null', 'w')
original_stdout = sys.stdout
# sys.stdout = dev_null

square_length = lambda  G, t  : euclidean_distance(G.nodes[0]['pos'], G.nodes[2]['pos'])

def get_theo_phase_shift(omega):
    

    ke=const.mu_apical
    kv = ke*60
    eta = const.eta
    kappa = eta + 2*kv

    arg = (kappa*ke/(omega*kv**2)+omega*eta/ke)/2
    delta = -np.arctan(arg)

    return delta

def get_phase_shift(omega, n_avg=3):

    path = './data/viscoelastic/'
    pattern = f'periodic_force_{f}_freq_{omega}_*.pickle'
    start_timestamp = get_creationtime( pattern.replace("*","0"), path=path)
    files  = get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp)
    final_time = files[-1][1]

    T = 2*np.pi/omega

    start_time = 0
    for i in range(len(files)-1,0,-1):
        if files[i][1]<final_time-(n_avg+2)*T:
            start_time=files[i][1]
            break

    print(f'loading omega: {omega}')
    res = analyze_network_evolution(path=path, start_time=start_time,
                            pattern=pattern,
                            func=square_length)
    res = np.array(res)
    t=res[:,0]

    numer_peaks = find_peaks(res[:,1])[0]
    theo_peaks = find_peaks(np.sin(omega*t))[0]

    dt = (np.mean(np.diff(t)))
    time_shift = -np.mean(np.abs(theo_peaks[-3:]-numer_peaks[-3:]))*dt
    print(f'done omega: {omega}')
    return time_shift*omega

    


if __name__ == '__main__':

    pool = Pool(nodes=4)
    omega=omegas[::6]

    shifts = list(map( get_phase_shift, omegas ))
    theo_shifts = list(map( get_theo_phase_shift, omegas ))
    linewidth=3
    fig=plt.figure()
    plt.plot(omegas, shifts, '--',linewidth=linewidth, label='numerical', marker='o')
    plt.plot(omegas, theo_shifts, '--',linewidth=linewidth, label='theoretical')
    plt.xlabel(r"$\omega$ (radians s$^{-1}$)", fontsize=14)
    plt.ylabel('phase shift (radians)', fontsize=14)
    plt.legend()
    plt.savefig('./Validation/Viscoelastic/viscoelastic_phase_shift.png', dpi=300)
    plt.show(block=True)

    print('All Done!')
