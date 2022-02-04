import numpy as np
from numba.typed import List

from .util import new_graph
from .funcs import *
from . import globals as const

def square_grid_2d(N,M, h=const.default_edge['l_rest'], embed=3):


    G = new_graph()
    nodes=[]
    edges=[]
    i,j=np.mgrid[0:N,0:M]
    i=i.ravel()
    j=j.ravel()
    p0=np.vstack((i,j))
    p1=np.vstack((i+1,j))
    p2=np.vstack((i,j+1))
    p3=np.vstack((i+1,j+1))
    
    neighbours = np.vstack((p0,p1,
                            p0,p2,
                            p1,p3,
                            p2,p3))


    shape = (N+1,M+1)

    i_edges= np.vstack((*neighbours[0::2],)).T.ravel()
    j_edges= np.vstack((*neighbours[1::2],)).T.ravel()

    edges = np.ravel_multi_index((i_edges, j_edges), shape).reshape(-1,2)
    edges = np.unique(edges, axis=0)

    ptot=np.unique(np.hstack((p0,p1,p2,p3)),axis=-1)

    pos_xy = ptot.T*h
    pos_xy -= np.mean(pos_xy,axis=0)

    inds = np.ravel_multi_index((*ptot,), shape)
    nodes=[(ind, {'pos':np.pad(pos,(0,embed-2))})for ind, pos in zip(inds, pos_xy)]
    G.add_nodes_from(nodes)

    G.add_edges_from([tuple(e) for e in edges], **const.default_edge)
    return G




def tissue_3d(hex=7, spoke_attr = const.default_edge, cell_edge_attr = const.default_edge, linker_attr = const.default_ab_linker):

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
        G.add_edges_from(spokes, **spoke_attr)

        if nx.__version__>"2.3":
            nx.classes.function.add_path(G,boundary, **cell_edge_attr)
        else:
            G.add_path(boundary, **cell_edge_attr)

        return

    G = new_graph()


    r = const.l_apical              # initial spoke length
    num_cells = 2*hex-1          # number of cells in center row

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

    for index in range(1,hex):  
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
   
    
    num_apical_nodes = i

    G.graph['num_apical_nodes']=num_apical_nodes
    G.graph['centers'] = centers
    # G.graph['circum_corted'] = circum_sorted

    # circum_sorted = []
    pos = np.array([*nx.get_node_attributes(G,'pos').values()])
    # xy = pos[:,0:2]
    # for center in centers:
    #     a, b = sort_corners(list(G.neighbors(center)),xy[center],xy)
    #     circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))
    # circum_sorted = np.array(circum_sorted)

    circum_sorted = get_circum_sorted(G, pos, centers)
    G.graph['circum_sorted']=circum_sorted

    # belt = []
    # for node in G.nodes():
    #     if len(list(G.neighbors(node))) < 6:
    #         belt.append(node)
    # xy_belt = [xy[n] for n in belt]
    # _,b = sort_corners(belt, xy[0], xy)
    # belt = np.array([b[n][0] for n in range(len(b))])
    
    # # make swaps for the "corner" nodes that the angle doesn't account for
    # for n in range(1,len(belt)-1):
    #     if G.has_edge(belt[n-1],belt[n]) == False:
    #         belt[n], belt[n+1] = belt[n+1], belt[n]

    belt = get_outer_belt(G)
    # triangles = get_triangles(G, pos, centers, belt)
    triangles = []
    for node in G.nodes():
        if node not in belt:
            if node in centers:
                out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
                alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,6)]

                for entry in alpha_beta:
                    triangles.append(entry)
            else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
	            
                for k in range(0,6):
                    alpha = [node,neighbors[k-1],neighbors[k-2]]
                    beta = [node,neighbors[k],neighbors[k-1]]
                    
                    if set(alpha) & set(centers) != set(beta) & set(centers):
                        triangles.append([alpha,beta])

    print("Apical nodes added correctly.")
    print("Number of apical nodes are", i)
    
    G_apical = new_graph(G)

    
    
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

    for index in range(1,hex):  
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
            G.add_edge(n, n+basal_offset, **linker_attr)

    print("Lateral Connections made")


    
    return G, G_apical


def get_outer_belt(G):
    if 'num_apical_nodes' in G.graph.keys():
        num_apical_nodes=G.graph['num_apical_nodes']
        pos = np.array([v for k,v in nx.get_node_attributes(G,'pos').items() if k<num_apical_nodes])
        xy = pos[:,0:2]
        belt = []
        for node in G.nodes():
            if node<num_apical_nodes and len(list(G.neighbors(node))) < 6:
                belt.append(node)
        # xy_belt = [xy[n] for n in belt]
        _, b = sort_corners(belt, xy[0], xy)
        belt = np.array([b[n][0] for n in range(len(b))])
        
        # make swaps for the "corner" nodes that the angle doesn't account for
        for n in range(1,len(belt)-1):
            if G.has_edge(belt[n-1],belt[n]) == False:
                belt[n], belt[n+1] = belt[n+1], belt[n]
    else:
        belt=None
    
    return belt

def get_triangles(G, pos, centers, belt):
    if belt is None:
        belt=get_outer_belt(G)
    
    if centers is None and 'centers' in G.graph.keys():
        centers = G.graph['centers']

    if belt is not None and centers is not None:
        centers=G.graph['centers']
        triangles = []
        for node in G.nodes():
            if node not in belt:
                if node in centers:
                    out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                    neighbors = [out2[k][0] for k in range(0,len(out2))]
                    alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,len(neighbors))]

                    for entry in alpha_beta:
                        triangles.append(entry)
                else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                    out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                    neighbors = [out2[k][0] for k in range(0,len(out2))]
                    
                    for k in range(0,6):
                        alpha = [node,neighbors[k-1],neighbors[k-2]]
                        beta = [node,neighbors[k],neighbors[k-1]]
                        
                        if set(alpha) & set(centers) != set(beta) & set(centers):
                            triangles.append([alpha,beta])
    else:
        triangles=None

    return triangles

def get_circum_sorted(G, pos, centers):
    circum_sorted = []
    xy = pos[:,0:2]
    for center in centers:
        a, b = sort_corners(list(G.neighbors(center)), xy[center],xy)
        circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))
    
    return circum_sorted



def new_topology(K, inter, cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes, linker_attr=None, edge_attr=None):
    # obtain new network topology - i.e. triangles, and circum_sorted 
    # inputs:   K: networkx graph (apical nodes only)
    #           inter: a python list of the nodes that have been intercalated  
    #           cents:
    #           temp1
    #           temp2
    #           belt
    #           centers
    #
    # returns:  circum_sorted - the peripheal nodes of the centers sorted. (update to previous)
    #           triangles - a numpy.array of the triangle pairs (update to previous)
    #           K - the new networkx Graph preserving the topology

    l_mvmt = const.l_mvmt
    node = inter[0]
    neighbor = inter[1]
    pos = nx.get_node_attributes(K,'pos')
    a = pos[node]
    b = pos[neighbor]
    
    # collapse nodes to same position 
    K.node[node]['pos'] = np.array([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0])
    K.node[neighbor]['pos'] = np.array([(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0])

    # move nodes toward new center 
    mvmt = unit_vector(a,pos[cents[1]])
    K.node[node]['pos'] = np.array([a[0]+l_mvmt*mvmt[0], a[1]+l_mvmt*mvmt[1], a[2]+l_mvmt*mvmt[2]])
    mvmt = unit_vector(b,pos[cents[0]])
    K.node[neighbor]['pos'] = np.array([b[0]+l_mvmt*mvmt[0], b[1]+l_mvmt*mvmt[1], b[2]+l_mvmt*mvmt[2]])

    # sever connections
    K.remove_edge(node,cents[0])
    K.remove_edge(node,temp1[0])
    K.remove_edge(neighbor,cents[1])
    K.remove_edge(neighbor,temp2[0])

    # add new connections
    # new edges 
    K.add_edge(node,temp2[0],myosin=0,color='#808080')
    K.add_edge(neighbor,temp1[0],myosin=0,color='#808080')
    # new spokes 
    K.add_edge(neighbor,ii,l_rest = const.l_apical, myosin=0)
    K.add_edge(node,jj,l_rest = const.l_apical, myosin=0)
    
    # new network made. Now get circum_sorted
    # update pos list 
    circum_sorted = [] 
    pos = np.array([*nx.get_node_attributes(K,'pos').values()])
    xy = np.array([np.array(pos[n]) for n in range(0,num_api_nodes)])
    
    # be safe, just sort them all over again 
    for center in centers:
        a, b = sort_corners(list(K.neighbors(center)),xy[center],xy)
        circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))

    
    circum_sorted = get_circum_sorted(K, pos, centers)
    triangles = get_triangles(K, pos, centers, belt)

    # triangles = []
    # for node in K.nodes():
    #     if node not in belt:
    #         if node in centers:
    #             out1, out2 = sort_corners(list(K.neighbors(node)),pos[node],pos)
    #             neighbors = [out2[k][0] for k in range(0,len(out2))]
    #             alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,len(neighbors))]

    #             for entry in alpha_beta:
    #                 triangles.append(entry)
    #         else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
    #             out1, out2 = sort_corners(list(K.neighbors(node)),pos[node],pos)
    #             neighbors = [out2[k][0] for k in range(0,len(out2))]
	            
    #             for k in range(0,len(neighbors)):
    #                 alpha = [node,neighbors[k-1],neighbors[k-2]]
    #                 beta = [node,neighbors[k],neighbors[k-1]]
                    
    #                 if set(alpha) & set(centers) != set(beta) & set(centers):
    #                     triangles.append([alpha,beta])
        
    return circum_sorted, triangles, K



