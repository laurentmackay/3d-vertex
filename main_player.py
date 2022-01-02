from numpy.core.numeric import Infinity

from tissue_3d import tissue_3d
from vertex_3d import *
import SG
from pyqt_viz import *
from globals import inter_edges

if __name__ == '__main__':

    # initialize the tissue
    G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()

    #initialize some things for the callback
    invagination = SG.invagination(G, belt, centers)

    player = network_player(G, attr='myosin', cell_edges_only=True, apical_only=True)
    
    while True:
        pass
