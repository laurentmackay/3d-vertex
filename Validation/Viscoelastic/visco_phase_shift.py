from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import vertex_integrator
from VertexTissue.Tissue import square_grid_2d
import VertexTissue.SG as SG
from VertexTissue.funcs import unit_vector_2D
from VertexTissue.globals import default_ab_linker, default_edge


N=1
M=1

omegas =( *np.logspace(-1, -2, num= 20),)
omegas = ( *np.geomspace(0.5, 0.125, num= 8),) + omegas 
f=1

def main(omega):

    

    G = square_grid_2d( N, M)
    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ *l1 , *l2]
    pos_a = G.nodes[forced_points[0]]['pos']
    pos_b = G.nodes[forced_points[-(M+1)]]['pos']

    force_vec = f*unit_vector_2D(pos_a, pos_b)

    for a,b in zip(l1,l2):
        G[a][b]['tau'] = 60

    # forced_points=[range(N+1)]

    forces=[-force_vec,  -force_vec,   force_vec,  force_vec]   

    def forcing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f*np.sin(t*omega)



    #create integrator
    integrate = vertex_integrator(G, G, pre_callback=forcing, ndim=2, save_rate=0.1, maxwell=True)
    #integrate
        
    T=2*np.pi/omega

    #integrate
    try:
        print(f'Starting: omega={omega}', file=original_stdout)
        integrate(0.01, 40*T, save_pattern=None)
        integrate(0.01, 10*T, save_pattern='data/viscoelastic/'+f'periodic_force_{f}_freq_{omega}_*.pickle')
        print(f'Done: omega={omega}', file=original_stdout)

    except Exception as e:

        print(f'###############\n###############\n failed to integrate tau={omega} \n {e} \n###############\n###############', file=original_stdout)

        pass
    


if __name__ == '__main__':

    dev_null = open('/dev/null', 'w')
    original_stdout = sys.stdout
    sys.stdout = dev_null


    pool = Pool(nodes=6)

    

    pool.map( main, omegas )

    print('All Done!')
