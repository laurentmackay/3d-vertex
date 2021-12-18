import networkx as nx
import numpy as np

import globals as const
from funcs import *

###############
def tissue_3d():

    def gen_nodes(ori,z):
        nodes = [[ori[0] + r*np.cos(n*np.pi/3), ori[1] + r*np.sin(n*np.pi/3),z] for n in range(0,6)]
        return np.array(nodes)

    def add_nodes(nodes, i):
        pos = nx.get_node_attributes(G,'pos')
        cen_index = i-1
        if i < 1000:
            centers.append(cen_index)
        AS_boundary = []
        spokes = []
        for node in nodes:
            add_node = True
            for existing_node in pos:
                if euclidean_distance(pos[existing_node],node) < 10**(-7):
                    add_node = False
                    AS_boundary.append(existing_node)
                    spokes.append((cen_index,existing_node))
                    break

            if add_node == True:
                G.add_node(i,pos=node)
                i += 1
                AS_boundary.append(i-1)
                spokes.append((cen_index,i-1))

        return AS_boundary, spokes, i

    def add_spokes_edges(spokes, boundary):
        boundary.append(boundary[0])
        G.add_edges_from(spokes, l_rest=const.l_apical, myosin=0)
        attr = {'l_rest' : const.l_apical, 'myosin':0, 'color':'#808080'}
        if nx.__version__>"2.3":
            nx.classes.function.add_path(G,boundary, **attr)
        else:
            G.add_path(boundary, **attr)

        return

    G = nx.Graph()
    if nx.__version__>"2.3":
        G.node=G._node

    r = const.l_apical              # initial spoke length
    num_cells = 2*const.hex-1          # number of cells in center row

    centers = []
    
    # Apical Nodes
    i = 0
    # Center cell set up
    z = 0.0
    origin = [0.0,0.0,z]
    G.add_node(i,pos=np.array(origin))
    i += 1

    nodes = gen_nodes(origin,z)
    AS_boundary, spokes, i = add_nodes(nodes,i)
    add_spokes_edges(spokes, AS_boundary)

    for index in range(1,int((num_cells - 1)/2.)+1):
        # # Step Up
        origin = [0, np.sqrt(3)*r*index,0.0]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

        # # # Step down
        origin = [0, -np.sqrt(3)*r*index,0.0]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

    for index in range(1,const.hex):  
        if (num_cells - index) % 2 == 0:
            for j in range(1,(num_cells-index),2):
                origin = [(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

            # Step Left

                origin = [-(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [-(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

        else:
            for j in range(0,(num_cells-index),2):
                origin = [3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

                # Step Left
                origin = [-3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [-3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)
   
    circum_sorted = []
    pos = nx.get_node_attributes(G,'pos') 
    xy = np.array([[pos[n][0],pos[n][1]] for n in range(0,i)])
    for center in centers:
        a, b = sort_corners(list(G.neighbors(center)),xy[center],xy)
        circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))
    circum_sorted = np.array(circum_sorted)

    belt = []
    for node in G.nodes():
        if len(list(G.neighbors(node))) < 6:
            belt.append(node)
    xy_belt = [xy[n] for n in belt]
    a,b = sort_corners(belt, xy[0], xy)
    belt = np.array([b[n][0] for n in range(len(b))])
    
    # make swaps for the "corner" nodes that the angle doesn't account for
    for n in range(1,len(belt)-1):
        if G.has_edge(belt[n-1],belt[n]) == False:
            belt[n], belt[n+1] = belt[n+1], belt[n]

    triangles = []
    for node in G.nodes():
        if node not in belt:
            if node in centers:
                _, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
                alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,6)]

                for entry in alpha_beta:
                    triangles.append(entry)
            else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                _, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
	            
                for k in range(0,6):
                    alpha = [node,neighbors[k-1],neighbors[k-2]]
                    beta = [node,neighbors[k],neighbors[k-1]]
                    
                    if set(alpha) & set(centers) != set(beta) & set(centers):
                        triangles.append([alpha,beta])

    print(f'Apical nodes added correctly.\nNumber of apical nodes: {i}')

    
    G2D = G.copy()
    if nx.__version__>"2.3":
        G2D.node = G2D._node

    num_apical_nodes = i
    
    # Basal Nodes
    i = 1000
    z = -const.l_depth
    # Center cell set up
    origin = [0.0,0.0,z]
    G.add_node(i,pos=np.array(origin))
    i += 1

    nodes = gen_nodes(origin,z)
    AS_boundary, spokes, i = add_nodes(nodes,i)
    add_spokes_edges(spokes, AS_boundary)

    for index in range(1,int((num_cells - 1)/2.)+1):
        # # Step Up
        origin = [0, np.sqrt(3)*r*index,z]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

        # # # Step down
        origin = [0, -np.sqrt(3)*r*index,z]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

    for index in range(1,const.hex):  
        if (num_cells - index) % 2 == 0:
            for j in range(1,(num_cells-index),2):
                origin = [(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

            # Step Left

                origin = [-(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [-(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

        else:
            for j in range(0,(num_cells-index),2):
                origin = [3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

                # Step Left
                origin = [-3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [-3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

    print("Basal Nodes Added")
    for n in range(0,num_apical_nodes):
        if n not in centers:
            G.add_edge(n,n+basal_offset, l_rest = const.l_depth, myosin = 0, beta = 0)

    print("Lateral Connections made")
    
    return G, G2D, centers, num_apical_nodes, circum_sorted, belt, triangles