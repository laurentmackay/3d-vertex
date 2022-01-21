from vertex_3d import *
import SG
from PyQtViz import edge_viewer


if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    invagination = SG.arcs_with_intercalation(G, belt)
    viewer = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)
    t_last = 0 
    t_plot = 1



    #create integrator
    integrate = vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=invagination, ndim=3)
    #integrate
    integrate(0.5,2000, save_pattern='data/testing/elastic_*.pickle')
