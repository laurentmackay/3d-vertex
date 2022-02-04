from funcs import unit_vector_2D
from vertex_3d import vertex_integrator
from Tissue import square_grid_2d
import numpy as np
from PyQtViz import edge_viewer

if __name__ == '__main__':

    # initialize the tissue
    N=1
    M=1
    G = square_grid_2d( N, M)
    l1 = list(range(M+1))
    l2 = list(range(len(G)-M-1, len(G)))
    forced_points = [ *l1 , *l2]
    pos_a = G.nodes[forced_points[0]]['pos']
    pos_b = G.nodes[forced_points[-(M+1)]]['pos']

    force_vec = 1*unit_vector_2D(pos_a, pos_b)

    # forced_points=[range(N+1)]

    forces=[-force_vec,  -force_vec,   force_vec,  force_vec]

    def forcing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f*np.sin(t/5)



    #create integrator
    integrate = vertex_integrator(G, G, pre_callback=forcing, ndim=2, player=True, save_rate=1)
    #integrate
    integrate(0.1, 2000, save_pattern='data/elastic/extension_periodic_*.pickle')
    print('Done')