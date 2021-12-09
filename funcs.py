##########
#
# funcs.py
#
#
# Author: Clinton H. Durney
# Email: cdurney@math.ubc.ca
#
# Last Edit: 11/8/19
##########

import networkx as nx
from scipy.spatial import distance
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import numpy as np
import globals as const
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
import sys
from numba import jit
import math
basal_offset = const.basal_offset

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
    
    G2D = G.copy()
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
###############

def vector(A,B):
        
    return [(B[0]-A[0]), (B[1]-A[1]), (B[2]-A[2])] 

@jit(nopython=True)
def euclidean_distance(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist

@jit(nopython=True)
def unit_vector(A,B):
    # Calculate the unit vector from A to B in 3D

    dist = euclidean_distance(A,B)

    if dist < 10e-15:
        dist = 1.0

    return (B-A)/dist
###############
# @jit(nopython=True)
def unit_vector_and_dist(A,B):
    # Calculate the unit vector from A to B in 3D

    dist = euclidean_distance(A,B)

    if dist < 10e-15:
        dist = 1.0

    return (B-A)/dist, dist
###############

def unit_vector_2D(A,B):
    # Calculate the unit vector from A to B in 3D

    dist = euclidean_distance(A,B)

    if dist < 10e-15:
        dist = 1.0

    return (B-A)[0:2]/dist
###############

# def d_pos(position,force,dt):
#     # Calculate the new vertex position using forward euler
#     # input: Current position, force, and dt the time step
#     # output: Updated position of the node.  

#     x_new = position[0] + (dt/const.eta)*force[0]

#     y_new = position[1] + (dt/const.eta)*force[1]

#     z_new = position[2] + (dt/const.eta)*force[2]
    
#     return [x_new,y_new,z_new]
###############

def elastic_force(l,l0,muu):
    # Calculate the magnitude of the force obeying Hooke's Law

    frce = muu*(l-l0) 

    return frce 
###############

def get_angle_formed_by(p1,p2,p3): # angle formed by three positions in space

    # based on code submitted by Paul Sherwood
    r1 = np.linalg.norm([p1[0]-p2[0],p1[1]-p2[1]])
    r2 = np.linalg.norm([p2[0]-p3[0],p2[1]-p3[1]])
    r3 = np.linalg.norm([p1[0]-p3[0],p1[1]-p3[1]])
                                        
    small = 1.0e-10
                                         
    if (r1 + r2 - r3) < small:
    # This seems to happen occasionally for 180 angles 
        theta = np.pi
    else:
        theta = np.arccos( (r1*r1 + r2*r2  - r3*r3) / (2.0 * r1*r2) )
    
    return theta;
###############

def signed_angle(v1,v2):
    theta = np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])
    if theta > np.pi:
        theta -= 2*np.pi
    elif theta <= -np.pi:
        theta += 2*np.pi
    return theta
###############

def tetrahedron_volume(a, b, c, d):
    
    return np.abs(np.einsum('ij,ij->i', a-d, crossMatMat(b-d, c-d))) / 6

def convex_hull_volume(pts):

    ch = ConvexHull(pts)
    dt = Delaunay(pts[ch.vertices])
    tets = dt.points[dt.simplices]

    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))

def convex_hull_volume_bis(pts):

    ch = ConvexHull(pts)
    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex), ch.simplices))
    tets = ch.points[simplices]

    return np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def get_points(G, q, pos):
    # get node numbers associated with a given center
    # inputs:   G: networkx graph
    #           q: number of center node (apical only)
    #           pos: position of nodes 
    # returns:  pts: list of positions that are associated with that center 

    api_nodes = [q] + list(G.neighbors(q))
    basal_nodes = [q+basal_offset] + list(G.neighbors(q+basal_offset)) 
#    basal_nodes = [api_nodes[n] + 1000 for n in range(1,7)]
    pts = api_nodes + basal_nodes
    pts = [pos[n] for n in pts]

    return pts 

@jit(nopython=True)
def cross33(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@jit(nopython=True)
def cross3Mat(a,b):
    out = np.zeros((b.shape))
    for i in range(0,b.shape[0]):
        out[i,0]=a[1]*b[i,2]-a[2]*b[i,1]
        out[i,1]=a[2]*b[i,0]-a[0]*b[i,2]
        out[i,2]=a[0]*b[i,1]-a[1]*b[i,0]

    return out

@jit(nopython=True)
def crossMatMat(a,b):
    out = np.zeros((b.shape))
    for i in range(0,b.shape[0]):
        out[i,0]=a[i,1]*b[i,2]-a[i,2]*b[i,1]
        out[i,1]=a[i,2]*b[i,0]-a[i,0]*b[i,2]
        out[i,2]=a[i,0]*b[i,1]-a[i,1]*b[i,0]

    return out    



def sort_corners(corners,center_pos,pos_nodes):

    corn_sort = [(corners[0],0)]
    u = unit_vector_2D(center_pos,pos_nodes[corners[0]])
    for i in range(1,len(corners)):
        v = unit_vector_2D(center_pos,pos_nodes[corners[i]])
        dot = np.dot(u,v)
        det = np.linalg.det([u,v])
        angle = np.arctan2(det,dot)
        corn_sort.append((corners[i],angle))
        corn_sort = sorted(corn_sort, key=lambda tup: tup[1])
        corn2 = [pos_nodes[entry[0]] for entry in corn_sort]
    
    return corn2, corn_sort

@jit(nopython=True)
def area_side(pos_side):
    
    A_alpha = np.zeros((3,))
    # inds=[2,0,1]
    for i in range(0,3):
        A_alpha += (1/2)*cross33(pos_side[i],pos_side[i-1])
    
    return np.linalg.norm(A_alpha), A_alpha


@jit(nopython=True)
def be_area_2( pos_alpha, pos_beta):
    
    A_alpha = np.zeros((3,))
    A_beta = np.zeros((3,))
    # inds=np.array([2,0,1])
    for i in range(0,3):
        A_alpha += (1/2)*cross33(pos_alpha[i],pos_alpha[i-1])
    
        A_beta += (1/2)*cross33(pos_beta[i],pos_beta[i-1])
    
    # A_alpha = np.sum(crossMatMat(pos_alpha,pos_alpha[inds]),axis=0)
    # A_beta = np.sum(crossMatMat(pos_beta,pos_beta[inds]),axis=0)
    return np.linalg.norm(A_alpha), A_alpha, np.linalg.norm(A_beta), A_beta



e = np.array([[1,0,0], [0,1,0], [0,0,1]])

#@profile
@jit(nopython=True)
def bending_energy_2(nbhrs_alpha, nbhrs_beta, alpha_vec, A_alpha, beta_vec, A_beta, pos_alpha_A, pos_alpha_B, pos_beta_A, pos_beta_B):
    
    
    
    
    # initialize the sums to zero
    # sums = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    # sums = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    # sums=[];

    if nbhrs_alpha != False:
        sum0=np.sum(cross3Mat(pos_alpha_B-pos_alpha_A,e)*beta_vec,axis=0)/2
        sum4=np.sum(cross3Mat(pos_alpha_B-pos_alpha_A,e)*alpha_vec,axis=0)/2
    else:
        sum0=np.zeros((3,))
        sum4=np.zeros((3,))

    if nbhrs_beta != False:
        sum1=np.sum(cross3Mat(pos_beta_B-pos_beta_A,e)*alpha_vec,axis=0)/2
        sum3=np.sum(cross3Mat(pos_beta_B-pos_beta_A,e)*beta_vec,axis=0)/2
    else:
        sum1=np.zeros((3,))
        sum3=np.zeros((3,))

    sum2=np.repeat(alpha_vec[0]*beta_vec[0]+alpha_vec[1]*beta_vec[1]+alpha_vec[2]*beta_vec[2],3)

    # sums2=np.vstack([np.sum(np.cross(pos[nbhrs_alpha[-1]]-pos[nbhrs_alpha[0]],e)*A_beta[1].reshape((-1,1)),axis=0)/2,
    # np.sum(np.cross(pos[nbhrs_beta[-1]]-pos[nbhrs_beta[0]],e)*A_alpha[1].reshape((-1,1)),axis=0)/2,
    # np.tile(np.sum(A_alpha[1]*A_beta[1]),reps=(3,)),
    # np.sum(np.cross(pos[nbhrs_beta[-1]]-pos[nbhrs_beta[0]],e)*A_beta[1].reshape((-1,1)),axis=0)/2,
    # np.sum(np.cross(pos[nbhrs_alpha[-1]]-pos[nbhrs_alpha[0]],e)*A_alpha[1].reshape((-1,1)),axis=0)/2])

    # for k in range(0,3):
    #     # sum (1) and (5) use the alpha cell
    #     if nbhrs_alpha != False:
    #         cross = np.cross(pos[nbhrs_alpha[-1]]-pos[nbhrs_alpha[0]],e[k])
    #         sums[0] += A_beta[1][k]*(1/2)*cross
    #         sums[4] += A_alpha[1][k]*(1/2)*cross

    #     # sum (2) and (4) use the beta cell
    #     if nbhrs_beta != False:
    #         cross = np.cross(pos[nbhrs_beta[-1]]-pos[nbhrs_beta[0]],e[k])
    #         sums[1] += A_alpha[1][k]*(1/2)*cross
    #         sums[3] += A_beta[1][k]*(1/2)*cross

    #     # sum (3)
    #     sums[2] += A_alpha[1][k]*A_beta[1][k]

    # if not np.all(np.array(sums2)==sums):
    #     print('ru roh')

    return (1/(A_alpha*A_beta))*(sum0+sum1) \
            + (-sum2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sum3 \
            +(A_beta/A_alpha)*sum4)

def new_topology(K, inter, cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes):
    # obtain new network topology - i.e. triangles, and circum_sorted 
    # inputs:   K: networkx graph
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
    K.node[node]['pos'] = [(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0]
    K.node[neighbor]['pos'] = [(a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0]

    # move nodes toward new center 
    mvmt = unit_vector(a,pos[cents[1]])
    K.node[node]['pos'] = [a[0]+l_mvmt*mvmt[0], a[1]+l_mvmt*mvmt[1], a[2]+l_mvmt*mvmt[2]]
    mvmt = unit_vector(b,pos[cents[0]])
    K.node[neighbor]['pos'] = [b[0]+l_mvmt*mvmt[0], b[1]+l_mvmt*mvmt[1], b[2]+l_mvmt*mvmt[2]]

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
    pos = nx.get_node_attributes(K,'pos')
    xy = [[pos[n][0],pos[n][1]] for n in range(0,num_api_nodes)]
    
    # be safe, just sort them all over again 
    for center in centers:
        a, b = sort_corners(list(K.neighbors(center)),xy[center],xy)
        circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))
    circum_sorted = np.array(circum_sorted)

    triangles = []
    for node in K.nodes():
        if node not in belt:
            if node in centers:
                out1, out2 = sort_corners(list(K.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
                alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,len(neighbors))]

                for entry in alpha_beta:
                    triangles.append(entry)
            else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                out1, out2 = sort_corners(list(K.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
	            
                for k in range(0,len(neighbors)):
                    alpha = [node,neighbors[k-1],neighbors[k-2]]
                    beta = [node,neighbors[k],neighbors[k-1]]
                    
                    if set(alpha) & set(centers) != set(beta) & set(centers):
                        triangles.append([alpha,beta])
        
    return circum_sorted, triangles, K



