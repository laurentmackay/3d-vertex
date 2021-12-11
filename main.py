from vertex_3d import *
import SG
from mlab_viz import edge_viewer


if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    invagination = SG.invagination(G, belt)
    viewer = edge_viewer(G,attr='myosin')
    t_last = 0 
    t_plot = 5

    def callback(t):
        invagination(t)
        if t-t_last>=t_plot:
            viewer(G)

    #create integrator
    integrate = vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=callback)
    #integrate
    integrate(0.5,2000)
