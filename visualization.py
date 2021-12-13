# %%
import networkx as nx
import numpy as np
import vtk
import matplotlib.pyplot as plt

from mayavi import mlab
mlab.init_notebook('x3d', 800, 800)

# %% [markdown]
# ## Current dictionary of code that is working.

# %%
# This works to visualize the nodes and edges.  
# Things to change/work on:  size of edges, size of nodes, coloring of the nodes, coloring of the edges by a 
# scalar - potentially myosin concentration.


# Visualize the data ##########################################################
# This one works!!!!!!!!!

mlab.figure(1, bgcolor=(0, 0, 0))
mlab.clf()# Visualize the data ##########################################################
# This one works!!!!!!!!!

mlab.figure(1, bgcolor=(0, 0, 0))
mlab.clf()

# pts = mlab.points3d(x, y, z, 1.5 * scalars.max() - scalars,
#                                     scale_factor=0.015, resolution=10)

G = nx.convert_node_labels_to_integers(G1)
pos = nx.get_node_attributes(G,'pos')
xyz = np.array([pos[v] for v in sorted(G)])
scalars = np.array(list(G.nodes())) + 5
pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    scalars,
                    scale_factor=0.1,
                    scale_mode='none',
                    colormap='Blues',
                    resolution=20)

pts.mlab_source.dataset.lines = np.array(list(G.edges()))

# Use a tube fiter to plot tubes on the link, varying the radius with the
# scalar value
tube = mlab.pipeline.tube(pts, tube_radius=0.15)
tube.filter.radius_factor = 1.
# tube.filter.vary_radius = 'vary_radius_by_scalar'
mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))

# Visualize the local atomic density
mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))

# pts = mlab.points3d(x, y, z, 1.5 * scalars.max() - scalars,
#                                     scale_factor=0.015, resolution=10)

G = nx.convert_node_labels_to_integers(G)
pos = nx.get_node_attributes(G,'pos')
xyz = np.array([pos[v] for v in sorted(G)])
scalars = np.array(list(G.nodes())) + 5
pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    scalars,
                    scale_factor=0.1,
                    scale_mode='none',
                    colormap='Blues',
                    resolution=20)

pts.mlab_source.dataset.lines = np.array(list(G.edges()))

# Use a tube fiter to plot tubes on the link, varying the radius with the
# scalar value
tube = mlab.pipeline.tube(pts, tube_radius=0.15)
tube.filter.radius_factor = 1.
# tube.filter.vary_radius = 'vary_radius_by_scalar'
mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))

# Visualize the local atomic density
mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))

# %%
# apical surface using delaunay triangulation 
X = xyz[:151,0]
Y = xyz[:151,1]
Z = xyz[:151,2]

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")

#Basal surface using delaunay triangulation
X = xyz[152:,0]
Y = xyz[152:,1]
Z = xyz[152:,2]

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")

# %%
G = nx.read_gpickle('/home/cdurney/fly-glands/bending/t0.pickle')
num_apical_nodes = 4

# %%
# Visualize the data ##########################################################
# This one works!!!!!!!!!

mlab.figure(1, bgcolor=(1, 1, 1))
mlab.clf()

# pts = mlab.points3d(x, y, z, 1.5 * scalars.max() - scalars,
#                                     scale_factor=0.015, resolution=10)

G1 = nx.convert_node_labels_to_integers(G)
pos = nx.get_node_attributes(G1,'pos')
xyz = np.array([pos[v] for v in sorted(G1)])
scalars = np.array(list(G1.nodes())) + 5
pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    scalars,
                    scale_factor=0.1,
                    scale_mode='none',
                    colormap='Blues',
                    resolution=20)

pts.mlab_source.dataset.lines = np.array(list(G1.edges()))

# Use a tube fiter to plot tubes on the link, varying the radius with the
# scalar value
tube = mlab.pipeline.tube(pts, tube_radius=0.2)
tube.filter.radius_factor = 1.
# tube.filter.vary_radius = 'vary_radius_by_scalar'
mlab.pipeline.surface(tube, color=(0, 0, 0))

# Visualize the local atomic density
mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))

X = xyz[:num_apical_nodes,0]
Y = xyz[:num_apical_nodes,1]
Z = xyz[:num_apical_nodes,2]

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(X, Y, Z, Z)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# # Basal surface
# X = xyz[num_apical_nodes:,0]
# Y = xyz[num_apical_nodes:,1]
# Z = xyz[num_apical_nodes:,2]

# # Define the points in 3D space
# # including color code based on Z coordinate.
# pts = mlab.points3d(X, Y, Z, Z)

# # Triangulate based on X, Y with Delaunay 2D algorithm.
# # Save resulting triangulation.
# mesh = mlab.pipeline.delaunay2d(pts)

# # Remove the point representation from the plot
# pts.remove()

# # Draw a surface based on the triangulation
# surf = mlab.pipeline.surface(mesh)

# # Simple plot.
# mlab.xlabel("x")
# mlab.ylabel("y")
# mlab.zlabel("z")

# %%
G = nx.read_gpickle('/home/cdurney/fly-glands/r7l1/t2000.pickle')
G[209][210]['myosin']

# %%
for n in range(0,1500):
    file_name = 't' + str(n) + '.pickle'
    G1 = nx.read_gpickle('/home/cdurney/fly-glands/3dviz/3hex/2inter_2be/' + file_name)
    
    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()

    # pts = mlab.points3d(x, y, z, 1.5 * scalars.max() - scalars,
    #                                     scale_factor=0.015, resolution=10)

    G = nx.convert_node_labels_to_integers(G1)
    pos = nx.get_node_attributes(G,'pos')
    xyz = np.array([pos[v] for v in sorted(G)])
    scalars = np.array(list(G.nodes())) + 5
    pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                        scalars,
                        scale_factor=0.1,
                        scale_mode='none',
                        colormap='Blues',
                        resolution=20)

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))

    # Use a tube fiter to plot tubes on the link, varying the radius with the
    # scalar value
    tube = mlab.pipeline.tube(pts, tube_radius=0.2)
    tube.filter.radius_factor = 1.
    # tube.filter.vary_radius = 'vary_radius_by_scalar'
    mlab.pipeline.surface(tube, color=(0, 0, 0))

    # Visualize the local atomic density
    mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(pts))

    X = xyz[:num_apical_nodes,0]
    Y = xyz[:num_apical_nodes,1]
    Z = xyz[:num_apical_nodes,2]

    # Define the points in 3D space
    # including color code based on Z coordinate.
    pts = mlab.points3d(X, Y, Z, Z)

    # Triangulate based on X, Y with Delaunay 2D algorithm.
    # Save resulting triangulation.
    mesh = mlab.pipeline.delaunay2d(pts)

    # Remove the point representation from the plot
    pts.remove()

    # Draw a surface based on the triangulation
    surf = mlab.pipeline.surface(mesh)

    # Simple plot.
    mlab.xlabel("x")
    mlab.ylabel("y")
    mlab.zlabel("z")

    # Basal surface
    X = xyz[num_apical_nodes:,0]
    Y = xyz[num_apical_nodes:,1]
    Z = xyz[num_apical_nodes:,2]

    # Define the points in 3D space
    # including color code based on Z coordinate.
    pts = mlab.points3d(X, Y, Z, Z)

    # Triangulate based on X, Y with Delaunay 2D algorithm.
    # Save resulting triangulation.
    mesh = mlab.pipeline.delaunay2d(pts)

    # Remove the point representation from the plot
    pts.remove()

    # Draw a surface based on the triangulation
    surf = mlab.pipeline.surface(mesh)

    mlab.savefig('/home/cdurney/fly-glands/3dviz/3hex/2inter_2be/tmp%03d.png'%n,size=(1000,1000))

# %%
for n in range(0,600):
    file_name = 't' + str(n) + '.pickle'
    G1 = nx.read_gpickle('/home/cdurney/fly-glands/3D/' + file_name)
    
    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()

    pos = nx.get_node_attributes(G1,'pos')
    xyz = np.array([pos[v] for v in sorted(G1)])

    pts = mlab.points3d(xyz[:num_apical_nodes, 0], xyz[:num_apical_nodes, 1], xyz[:num_apical_nodes, 2], scale_factor=0.1, scale_mode='none', colormap='Blues', resolution=20)

    valid = [i for i in list(G1.edges()) if i[0]<1000 and i[1]<1000]
    pts.mlab_source.dataset.lines = np.array(valid)

    # Use a tube fiter to plot tubes on the link, varying the radius with the
    # scalar value
    tube = mlab.pipeline.tube(pts, tube_radius=0.02)
    tube.filter.radius_factor = 1.
    mlab.pipeline.surface(tube, color=(0, 0, 0))

    mlab.savefig('/home/cdurney/fly-glands/3D/tmp2d%03d.png'%n,size=(1000,1000))

# %%
G = nx.read_gpickle('/home/cdurney/fly-glands/3D/t150.pickle')
num_apical_nodes = 73

# %%
for i in range(0,num_apical_nodes):
    a_nbhrs = list(G.neighbors(i))
    b_nbhrs = list(G.neighbors(i+1000))

    for k in range(0,len(a_nbhrs)):
        if (b_nbhrs[k] - a_nbhrs[k]) != 1000 and (b_nbhrs[k] - a_nbhrs[k]) != -1000:
            print("Problem!!!")
            print(i)

# %%
edges = list(G.edges)
for i in range(0,len(edges)):
    if edges[i][0] < 1000 and edges[i][1] < 1000:
        if (edges[i][0]+1000,edges[i][1]+1000) not in edges:
            print(False)

# %%
G1 = nx.read_gpickle('/home/cdurney/fly-glands/3D/t45.pickle')
    
mlab.figure(1, bgcolor=(1, 1, 1))
mlab.clf()

pos = nx.get_node_attributes(G1,'pos')
xyz = np.array([pos[v] for v in sorted(G1)])

pts = mlab.points3d(xyz[num_apical_nodes:, 0], xyz[num_apical_nodes:, 1], xyz[num_apical_nodes:, 2], scale_factor=0.1, scale_mode='none', colormap='Blues', resolution=20)

valid = [i for i in list(G1.edges()) if i[0]<1000 and i[1]<1000]
pts.mlab_source.dataset.lines = np.array(valid)

# Use a tube fiter to plot tubes on the link, varying the radius with the
# scalar value
tube = mlab.pipeline.tube(pts, tube_radius=0.01)
tube.filter.radius_factor = 1.
mlab.pipeline.surface(tube, color=(0, 0, 0))

# %%



