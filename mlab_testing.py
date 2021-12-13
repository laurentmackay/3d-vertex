import os
from mlab_viz import *
from mayavi import mlab
if __name__ == '__main__':
    G = nx.read_gpickle(os.path.join(os.getcwd(),f't0.pickle'))
    viewer = edge_viewer(G,  attr='myosin')

    for i in range(1,300,1):
        G = nx.read_gpickle(os.path.join(os.getcwd(),f't{i}.pickle'))
        viewer(G)