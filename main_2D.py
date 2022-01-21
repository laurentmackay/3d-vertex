from vertex_3d import vertex_integrator
from Tissue import square_grid_2d
import numpy as np

if __name__ == '__main__':

    # initialize the tissue
    G = square_grid_2d()

    forced_points = [0,1]
    forces=[np.array([0,1]),np.array([0,-1])]
    
    def frocing(t,force_dict):
        for p,f in zip(forced_points, forces):
            force_dict[p] += f



    #create integrator
    integrate = vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=forcing)
    #integrate
    integrate(0.5,2000, save_pattern='data/testing/elastic_*.pickle')
