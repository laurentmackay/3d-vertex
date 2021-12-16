from tissue_3d import tissue_3d
from vertex_3d import vertex_integrator
import SG
from pyqt_viz import *


if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    invagination = SG.invagination(G, belt)
    viewer = edge_viewer(G,exec=True, attr='myosin')

    t_plot = 1

    def mkcallback():
        t_last=0.0
        def callback(t):
            nonlocal t_last
            invagination(t)
            if t-t_last>=t_plot:
                viewer(G)
                t_last=t

        return callback

    #create integrator
    integrate = vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=mkcallback())
    #integrate
    integrate(0.5,2000)
