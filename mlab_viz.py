import networkx as nx
import numpy as np
from mayavi import mlab

def view_edges(G, tube_radius=0.15, colormap='Blues', attr=None, color=None, radius_factor = 3):

    pos = nx.get_node_attributes(G,'pos')

    x = []
    y = []
    z = []
    connections = []
    s = []
    

    for i, e in enumerate(G.edges()):
        

        edge_xyz = np.array([pos[e[0]],pos[e[1]]])

        x.append(edge_xyz[:, 0])
        y.append(edge_xyz[:, 1])
        z.append(edge_xyz[:, 2])
        connections.append(np.array([2*i,2*i+1]))
        if attr is not None:
            scalar = G.get_edge_data(*e)[attr]
        else:
            scalar = 0 
        s.extend((scalar,scalar))


    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    s = np.hstack(s)

    pts = mlab.points3d(x,y,z,s,
                        scale_factor=0.1,
                        scale_mode='none',
                        colormap=colormap,
                        resolution=20)


    pts.mlab_source.dataset.lines = np.vstack(connections)
    
    tube = mlab.pipeline.tube(pts, tube_radius=tube_radius)
    tube.filter.radius_factor = radius_factor
    tube.filter.vary_radius = 'vary_radius_by_scalar'
    mlab.pipeline.surface(tube, color=color)

    pts.remove()

    mlab.show()
    
