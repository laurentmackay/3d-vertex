import numpy as np
from numba.typed import List

from .util import new_graph
from .funcs import *
from . import globals as const

l_apical = const.l_apical 
l_depth = const.l_depth 

l0_apical = l_apical
l0_basal = l_apical 
l0_wall = l_depth 

mu_apical = const.mu_apical         
mu_basal = const.mu_basal          
mu_wall = const.mu_wall          
myo_beta = const.myo_beta 
eta = const.eta 
press_alpha = const.press_alpha 
l_mvmt = const.l_mvmt
basal_offset=const.basal_offset



def linear_grid(N, h=const.default_edge['l_rest'], embed=3):


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

    G.add_edges_from([tuple(e) for e in edges], **const.default_edge)
    return G

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
    G.graph['centers'] = center_nodes

    pos = np.array([*nx.get_node_attributes(G,'pos').values()])

    circum_sorted = get_circum_sorted(G, pos, center_nodes)
    G.graph['circum_sorted']=circum_sorted



    belt = get_outer_belt(G)
    # triangles = get_triangles(G, pos, centers, belt)
    triangles = []
    for node in G.nodes():
        if node not in belt:
            if node in center_nodes:
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
                    
                    if set(alpha) & set(center_nodes) != set(beta) & set(center_nodes):
                        triangles.append([alpha,beta])

    print("Apical nodes added correctly.")
    print("Number of apical nodes are", i)
    G.graph['triangles']=triangles
    
    G_apical = new_graph(G)

    
    # Basal Nodes
 
    if basal:

        # const.basal_offset=np.maximum(const.basal_offset, len(G))

        basal_centers = apical_centers + np.array((0,0,-const.l_depth))
        kws = {'node_generator': node_generator,'cell_edge_attr':cell_edge_attr, 'spoke_attr':spoke_attr}

        add_2D_primitive_to_graph(G, origin=basal_centers[0], index=const.basal_offset, **kws)

        for origin in basal_centers[1:]:
            add_2D_primitive_to_graph(G, origin=origin, **kws)

        print("Basal Nodes Added")


        for n in range(0,num_apical_nodes):
            if n not in center_nodes:
                G.add_edge(n, n+basal_offset, **linker_attr)

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
                    
                    for k in range(0,6):
                        alpha = [node,neighbors[k-1],neighbors[k-2]]
                        beta = [node,neighbors[k],neighbors[k-1]]
                        
                        if set(alpha) & set(centers) != set(beta) & set(centers):
                            triangles.append([alpha, beta])
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
    K.add_edge(neighbor, ii, l_rest = const.l_apical, myosin=0)
    K.add_edge(node, jj, l_rest = const.l_apical, myosin=0)
    
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


def Tissue_Forces(G=None, ndim=3, minimal=False):

    pos = None
    dists = None
    drx = None

    def zero_force_dict(G):
        return {node: np.zeros(ndim ,dtype=float) for node in G.nodes()}

    def compute_distances_and_directions(G=G):
        nonlocal   pos, dists, drx

        dists = {node: 0 for node in G.edges()} 
        drx ={node: np.zeros(ndim ,dtype=float) for node in G.edges()} 

        pos = get_pos(G)
        
        
        for e in G.edges():
            
            direction, dist = unit_vector_and_dist(pos[e[0]],pos[e[1]])
            dists[e] = dist
            drx[e] = direction

        return dists, drx
    
    def compute_rod_forces(force_dict=None, G=G, compute_distances = True):
        nonlocal pos, dists, drx
        
        if force_dict is None:
            force_dict = zero_force_dict(G)

        if compute_distances:
            compute_distances_and_directions(G=G)

        l_rest = nx.get_edge_attributes(G,'l_rest')
        myosin = nx.get_edge_attributes(G,'myosin')
        
        for  e in G.edges():

            magnitude = mu_apical*(dists[e] - l_rest[e])
            magnitude2 = myo_beta*myosin[e]
            force = (magnitude + magnitude2)*drx[e][:ndim]

            force_dict[e[0]] += force
            force_dict[e[1]] -= force

    def compute_tissue_forces_3D(force_dict=None, G=G, compute_distances= True, compute_pressure=True):
        nonlocal pos

        circum_sorted = G.graph['circum_sorted']
        triangles = G.graph['triangles']
        centers = G.graph['centers']

        if force_dict is None:
            force_dict = zero_force_dict(G)

        compute_rod_forces(force_dict, G=G, compute_distances = compute_distances)


        if compute_pressure:
        
            # pre-calculate magnitude of pressure
            # index of list corresponds to index of centers list
            PI = np.zeros(len(centers),dtype=float) 
            # eventually move to classes?
            for n in range(len(centers)):
                # get nodes for volume
                pts = get_points(G, centers[n], pos) 
                # calculate volume
                vol = convex_hull_volume_bis(pts)  
                # calculate pressure
                PI[n] = -press_alpha*(vol-const.v_0) 


            
            for center, pts, pressure in zip(centers, circum_sorted, PI):  
                for i in range(len(pts)):
                    for inds in ((center, pts[i], pts[i-1]),
                                (center+basal_offset, pts[i-1]+basal_offset, pts[i]+basal_offset)):
                        pos_face =np.array([pos[j] for j in inds])
                        _, area_vec, _, _ = be_area_2(pos_face,pos_face)                       

                        force = pressure*area_vec/3.0
                        force_dict[inds[0]] += force
                        force_dict[inds[1]] += force
                        force_dict[inds[2]] += force
        


            # pressure for side panels
            # loop through each cell
            for cell_nodes, pressure in zip(circum_sorted, PI):
                # loop through the faces
                for i in range(len(cell_nodes)):
                    pts_id = (cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+basal_offset, cell_nodes[i-1]+basal_offset)
                    pts_pos = np.array([pos[pts_id[ii]] for ii in range(4)])
                    # on each face, calculate the center
                    center = np.average(pts_pos,axis=0)
                    # loop through the 4 triangles that make the face
                    for ii in range(0,4):
                        pos_side = np.array([center, pts_pos[ii-1], pts_pos[ii]])
                        _, area_vec = area_side(pos_side) 
                        
                        force = pressure*area_vec/2.0
                        force_dict[pts_id[ii-1]] += force
                        force_dict[pts_id[ii]] += force
        
        # Implement bending energy
        # Loop through all alpha, beta pairs of triangles
        offset=0
        for pair in triangles:
            for offset in (0, basal_offset):
                alpha = [i+offset for i in pair[0]]
                beta = [i+offset for i in pair[1]]
                
                # Apical faces, calculate areas and cross-products z
                pos_alpha = np.array([pos[i] for i in alpha])
                pos_beta = np.array([pos[i] for i in beta])
                A_alpha, A_alpha_vec, A_beta, A_beta_vec = be_area_2(pos_alpha, pos_beta)

                for inda, node in enumerate(alpha):
                    # inda = alpha.index(node) 
                    nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
                    if node in beta:
                        indb = beta.index(node)
                        nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 

                        frce = const.c_ab * bending_energy_2(True, True,A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[-1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[-1]])
                    else:
                        frce = const.c_ab * bending_energy_2(True, False, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]], pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]])
                    
                    force_dict[node] += frce

                for indb, node in enumerate(beta):
                    # don't double count the shared nodes
                    nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                    if node not in alpha:
                        # frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
                        frce = const.c_ab*bending_energy_2(False, True, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_beta[0]], pos[nbhrs_beta[1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[1]])

                        force_dict[node] += frce





    if ndim == 3 and not minimal:
        compute_forces=compute_tissue_forces_3D
    elif ndim == 2 or minimal:
        compute_forces=compute_rod_forces


    compute_forces.compute_distances_and_directions = compute_distances_and_directions

    return compute_forces
