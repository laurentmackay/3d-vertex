from numpy.core.numeric import Infinity
from vertex_3d import *
import SG
from pyqt_viz import edge_viewer
from globals import inter_edges

if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    invagination = SG.invagination(G, belt, )
    viewer = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)
    t_last = 0 
    t_plot = 0

    def mkcallback():
        t_last=0.0
        def callback(t,t_prev=-Infinity):
            nonlocal t_last
            invagination(t,t_prev)
            if t-t_last>=t_plot:
                viewer(G)
                t_last=t
        return callback

    #create integrator
    integrate = vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt,
                                  triangles, 
                                  pre_callback=mkcallback(),
                                  intercalation_callback=lambda a, b: viewer(G),
                                  length_prec=.1)
    #integrate
    integrate(5,20000)
