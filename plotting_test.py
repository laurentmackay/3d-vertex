from vertex_3d import *
from tissue_3d import tissue_3d
import SG
from pyqt_viz import edge_view



if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    edge_view(G,attr='myosin', exec=True, cell_edges=True)

