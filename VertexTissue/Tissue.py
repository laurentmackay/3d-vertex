import networkx as nx
import numpy as np


from .util import new_graph
from .Geometry import *
from . import globals as const

# l_apical = const.l_apical 
# l_depth = const.l_depth 

# l0_apical = l_apical
# l0_basal = l_apical 
# l0_wall = l_depth 

       
mu_basal = const.mu_basal          
mu_wall = const.mu_wall          
myo_beta = const.myo_beta 
eta = const.eta 

l_mvmt = const.l_mvmt




def linear_grid(N, h=const.default_edge['l_rest'], embed=3, edge_attrs=const.default_edge):


    G = new_graph()
    nodes=[]
    edges=[]
    i=np.arange(N)
    i=i.ravel()
    # j=j.ravel()
    p0=i
    p1=i+1

    
    neighbours = np.vstack((p0,p1))


    shape = (N+1,)

    edges= np.vstack((*neighbours,)).T.ravel().reshape(-1,2)
    edges = np.unique(edges, axis=0)

    ptot=np.unique(np.hstack((p0,p1)),axis=-1)

    pos_x = ptot.T*h
    pos_x -= np.mean(pos_x,axis=0)

    inds = ptot
    nodes=[(ind, {'pos':np.pad((pos, ),(0,embed-1))})for ind, pos in zip(inds, pos_x)]
    G.add_nodes_from(nodes)

    G.add_edges_from([tuple(e) for e in edges], **edge_attrs)
    return G

def square_grid_2d(N,M, h=None, spokes=False, scale_spokes=True, half_spokes=False, edge_attrs=const.default_edge, embed=3):
    '''
    generates a NxM network of squares, with nodes as depicted  
    p4 is only present when spokes=True

    p2-------p3
    | \     / |
    |  \   /  |
    |   p4    |
    |  /   \  |
    | /     \ |
    p0--------p1

    '''

    if h is None:
        h= edge_attrs['l_rest']

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
    if spokes:
        imax = (N+1)*(M+1)
        p4 = np.arange(imax,imax+N*M)
    
    neighbours = np.vstack((p0,p1,
                            p0,p2,
                            p1,p3,
                            p2,p3))


    shape = (N+1,M+1)

    i_edges= np.vstack((*neighbours[0::2],)).T.ravel()
    j_edges= np.vstack((*neighbours[1::2],)).T.ravel()

    edges = np.ravel_multi_index((i_edges, j_edges), shape).reshape(-1,2)
    edges = np.unique(edges, axis=0)

    ptot, inverse = np.unique(np.hstack((p0,p3,p1,p2)),axis=-1, return_inverse=True)

    pos_xy = ptot.T*h

    if spokes:
        pos_xy = np.vstack((pos_xy, np.vstack((i+0.5,j+0.5)).T*h))

    pos_xy -= np.mean(pos_xy,axis=0)

    inds = np.ravel_multi_index((*ptot,), shape)

    if spokes:
        inds = np.concatenate((inds,p4))
        
        # edges = np.vstack((edges, ))


    nodes=[(ind, {'pos':np.pad(pos,(0,embed-2))})for ind, pos in zip(inds, pos_xy)]
    G.add_nodes_from(nodes)

    G.add_edges_from([tuple(e) for e in edges], **edge_attrs)

    if spokes:
        spoke_dict = edge_attrs.copy()
        if scale_spokes:
            spoke_dict['l_rest']/=np.sqrt(2)
        
        if half_spokes:
            # _, inverse = np.unique(np.hstack((p0,p3)),axis=-1, return_inverse=True)
            N_spokes = 2
        else:
            N_spokes = 4

        spoke_edges = np.vstack((tuple(np.vstack((inverse[k*N*M:(k+1)*N*M],p4)).T for k in range(N_spokes))))
        G.add_edges_from([tuple(e) for e in spoke_edges], **spoke_dict)

    return G

PI_BY_3 = np.pi/3
HEX_THETA = np.array([n*PI_BY_3 for n in range(0,6)])
def hex_nodes(ori, r=const.l_apical, embed=3):
    x = ori[0]+r*np.cos(HEX_THETA)
    y = ori[1]+r*np.sin(HEX_THETA)
    if embed==3:
        return np.vstack((x,y, np.repeat(ori[2],repeats = (6,)))).T
    elif embed==2:
        return np.vstack((x,y)).T


def add_cell_to_graph(G, node_positions, center_node, next_node=None, thresh=10**(-7), spoke_attr = const.default_edge, cell_edge_attr = const.default_edge):
    
    pos = nx.get_node_attributes(G,'pos')
    cell_boundary = []
    spokes = []
    added_nodes = []
    
    if next_node is None:
        next_node = list(G.nodes)[-1]+1

    for node_pos in node_positions: #loop over new node positions
            add_node = True
            for existing_node in pos: #check that the new nodes do not overlap with an existing node
                if euclidean_distance(pos[existing_node], node_pos) < thresh:
                    add_node = False
                    cell_boundary.append(existing_node)
                    spokes.append((center_node, existing_node))
                    break

            if add_node == True: #if there was no overlap, add the new node to the cell boundary and create a spoke for it
                added_nodes.append(next_node)
                G.add_node(next_node, pos=node_pos)
                cell_boundary.append(next_node)
                spokes.append((center_node,next_node))
                next_node += 1

    #now add the cell boundary and spokes to the graph
    cell_boundary.append(cell_boundary[0])
    G.add_edges_from(spokes, **spoke_attr)

    if nx.__version__>"2.3":
        nx.classes.function.add_path(G, cell_boundary, **cell_edge_attr)
    else:
        G.add_path(cell_boundary, **cell_edge_attr)

    return added_nodes



def add_2D_primitive_to_graph(G, origin=np.array((0,0,0)),  index=None, node_generator = hex_nodes, thresh=10**(-7), spoke_attr = const.default_edge, cell_edge_attr = const.default_edge, linker_attr = const.default_ab_linker, centers=None, **kw):
    if index is None:
        if len(G):
            index = list(G.nodes)[-1]+1
        else:
            index = 0


    G.add_node(index, pos=np.array(origin))

    node_positions = node_generator(origin, **kw)

    add_cell_to_graph(G, node_positions, index, cell_edge_attr=cell_edge_attr, spoke_attr=spoke_attr)

    if centers is not None:
        centers.append(index)
    


def hex_hex_grid(hex=7, r = const.l_apical ):
    num_cells = 2*hex - 1          # number of cells in center row
    z=0.0
    origin = [0.0, 0.0, z]
    centers=[origin, ]

    for index in range(1,int((num_cells - 1)/2.)+1):
        centers.extend([[0, np.sqrt(3)*r*index,z], # # Step Up
                [0, -np.sqrt(3)*r*index, z]]) # # Step down

    for index in range(1,hex):  
        if (num_cells - index) % 2 == 0:
            for j in range(1,(num_cells-index),2):
                centers.extend([[(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z],
                            [(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z],
                            [-(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z],
                            [-(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                            ])
        else:
            for j in range(0,(num_cells-index),2):
                centers.append([3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z])
                if j != 0:
                    centers.append([3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z])
                # Step Left
                centers.append([-3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z])
                if j != 0:
                    centers.append([-3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z])

    return np.array(centers)

def T1_minimal( r = const.l_apical):
    y = r*(np.sqrt(3))/2
    x = r*3/2
    return np.array([[-x, 0, 0],
                     [x, 0, 0],
                     [0, y, 0],
                     [0, -y, 0]
                     ])

def T1_shell( r = const.l_apical):
    y = r*(np.sqrt(3))/2
    x = r*3/2
    return np.array([[-x, 0, 0],
                     [x, 0, 0],
                     [0, y, 0],[0, 3*y, 0],
                     [0, -y, 0],[0, -3*y, 0],
                     [-2*x, y, 0],[-2*x, -y, 0],
                     [2*x, -y, 0],[2*x, y, 0],
                     [x, -2*y, 0],[x, 2*y, 0],
                     [-x, -2*y, 0],[-x, 2*y, 0],
                     ])                   

def T1_shell2( r = const.l_apical):
    y = r*(np.sqrt(3))/2
    x = r*3/2
    return np.array([[-x, 0, 0],
                     [x, 0, 0],
                     [0, y, 0],[0, 3*y, 0],
                     [0, -y, 0],[0, -3*y, 0],
                     [-2*x, y, 0],[-2*x, -y, 0],
                     [2*x, -y, 0],[2*x, y, 0],
                     [x, -2*y, 0],[x, 2*y, 0],
                     [-x, -2*y, 0],[-x, 2*y, 0],
                     ])      

def tissue_3d( gen_centers=hex_hex_grid, node_generator=hex_nodes, basal=True, spoke_attr = const.default_edge, cell_edge_attr = const.default_edge, linker_attr = const.default_ab_linker, **kw):

    next_node=0




    G = new_graph()


    center_nodes = []
    
    # Apical Nodes
    i = 0
    next_node
    # Center cell set up


    



    kws = {'node_generator': node_generator,'cell_edge_attr':cell_edge_attr, 'spoke_attr':spoke_attr, 'centers':center_nodes}

    apical_centers = gen_centers(**kw)

    for origin in gen_centers(**kw):
        add_2D_primitive_to_graph(G, origin=origin, **kws)



    i=len(G)  
    
    num_apical_nodes = len(G)

    G.graph['num_apical_nodes']=num_apical_nodes
    G.graph['centers'] = np.array(center_nodes)

    pos = np.array([*nx.get_node_attributes(G,'pos').values()])

    circum_sorted = get_circum_sorted(G, pos, center_nodes)
    G.graph['circum_sorted']=circum_sorted



    belt = get_outer_belt(G)
    triangles = get_triangles(G, pos, center_nodes, belt)
    G.graph['triangles']=triangles

    print("Apical nodes added correctly.")
    print("Number of apical nodes are", i)

    

    G_apical = new_graph(G)

    
    # Basal Nodes
 
    if basal:


        G.graph['basal_offset']=i
        # const.basal_offset=np.maximum(const.basal_offset, len(G))

        basal_centers = apical_centers + np.array((0,0,-const.l_depth))
        kws = {'node_generator': node_generator,'cell_edge_attr':cell_edge_attr, 'spoke_attr':spoke_attr}

        add_2D_primitive_to_graph(G, origin=basal_centers[0], index=i, **kws)

        for origin in basal_centers[1:]:
            add_2D_primitive_to_graph(G, origin=origin, **kws)

        print("Basal Nodes Added")


        for n in range(0,num_apical_nodes):
            if n not in center_nodes:
                G.add_edge(n, n+i, **linker_attr)

        print("Vertical Connections made")

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


def get_outer_belt(G):
    if 'num_apical_nodes' in G.graph.keys():
        num_apical_nodes=G.graph['num_apical_nodes']
        pos = np.array([v for k,v in nx.get_node_attributes(G,'pos').items() if k<num_apical_nodes])
        xy = pos[:,0:2]
        outer = []
        for node in G.nodes():
            if node<num_apical_nodes and len(list(G.neighbors(node))) < 6:
                outer.append(node)
        belt=[outer[0],]

        outer_set=set(outer)
        i=0
        while len(belt)<len(outer):
            nhbrs=set(list(G.neighbors(belt[i])))
            new =  list((nhbrs & outer_set) - set(belt))
            if len(new)==1 or len(belt)==1:
                new=new[0]
            elif len(new):
                new = new[np.argmin([np.sum([b in G.graph['centers']  for b in  list(G.neighbors(_))] ) for _ in new])] #get the node connect to the least cells
                
            belt.append(new)
            i+=1

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
                    alpha_beta = [[[node, neighbors[k-1], neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,len(neighbors))]

                    for entry in alpha_beta:
                        triangles.append(entry)
                else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                    out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                    neighbors = [out2[k][0] for k in range(0,len(out2))]
                    
                    for k in range(len(neighbors)):
                        alpha = [node,neighbors[k-1],neighbors[k-2]]
                        beta = [node,neighbors[k],neighbors[k-1]]
                        
                        if set(alpha) & set(centers) != set(beta) & set(centers):
                            triangles.append([alpha, beta])

        triangles = np.array(triangles)
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



def new_topology(K, inter, cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes, linker_attr=None, edge_attr=None, adjust_network_positions=False):
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
    #           K_centerless - the new networkx Graph preserving the topology, without center nodes

    l_mvmt = const.l_mvmt
    node = inter[0]
    neighbor = inter[1]
    
    
    if adjust_network_positions:
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
        K.add_edge(neighbor, ii, l_rest = const.l_apical, myosin=0)
        if len(jj):
            K.add_edge(node, jj[0], l_rest = const.l_apical, myosin=0)
    
    # new network made. Now get circum_sorted
    # update pos list 
    circum_sorted = [] 
    pos = np.array([*nx.get_node_attributes(K,'pos').values()])
    xy = np.array([np.array(pos[n]) for n in range(0,num_api_nodes)])
    
    # be safe, just sort them all over again 
    # for center in centers:
    #     a, b = sort_corners(list(K.neighbors(center)),xy[center],xy)
    #     circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))

    
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

