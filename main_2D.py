from funcs import unit_vector_2D
from vertex_3d import vertex_integrator
from Tissue import square_grid_2d
import numpy as np
from PyQtViz import edge_viewer

if __name__ == '__main__':

    # initialize the tissue
    N=2
    M=3
    G = square_grid_2d(2,3)

    forced_points = [0, len(G)-1]
    pos_a = G.nodes[forced_points[0]]['pos']
    pos_b = G.nodes[forced_points[-1]]['pos']

    force_vec = unit_vector_2D(pos_a, pos_b)

    # forced_points=[range(N+1)]

    forces=[-force_vec, force_vec]

    def forcing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f



    #create integrator
    integrate = vertex_integrator(G, G, pre_callback=forcing, ndim=2, player=True)
    #integrate
    integrate(0.1, 20000, save_pattern='data/testing/elastic_*.pickle')
    print('Done')