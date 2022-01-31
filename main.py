from vertex_3d import vertex_integrator
from Tissue import tissue_3d, get_outer_belt
import SG
from PyQtViz import edge_viewer


if __name__ == '__main__':

    # initialize the tissue
    G, G_apical = tissue_3d()
    belt = get_outer_belt(G_apical)

    #initialize some things for the callback
    # invagination = SG.arcs_with_intercalation(G, belt)
    invagination = SG.just_arcs(G, belt)
    # viewer = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)
    t_last = 0 
    t_plot = 1



    #create integrator
    integrate = vertex_integrator(G, G_apical, pre_callback=invagination, player=True)
    #integrate
    integrate(0.5,2000, save_rate=1, save_pattern='data/testing/elastic_*.pickle')
