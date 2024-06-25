import networkx as nx
import numpy as np
import itertools
from scipy.spatial import distance
from VertexTissue.Geometry import euclidean_distance



import VertexTissue.globals as const
from  VertexTissue.funcs_orig import *

import time

# Constants for simulation
dt = const.dt
#var_dt = True

# dimensions of the cell 
l_apical = const.l_apical 
l_depth = const.l_depth 

# Set the arcs list
inner_arc = const.inner_arc
outer_arc = const.outer_arc

# mechanical parameters
# rest lengths of the passively elastic apical, basal and cell walls
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

# initialize the tissue
# G, K, centers, num_api_nodes, circum_sorted, belt, triangles = tissue_3d()
# pit_centers = const.pit_centers 

# viewer = edge_viewer(G,attr='myosin')

# t_plot=5
# t_last=-t_plot

def compute_forces_orig(G):

    centers = G.graph['centers']
    circum_sorted = G.graph['circum_sorted']
    basal_offset =  G.graph['basal_offset']
    triangles = G.graph['triangles']
    # triangles = triangles[int(triangles.shape[0]/2):,:,:]
    
    pos = nx.get_node_attributes(G,'pos')

    forces = np.zeros((len(G.nodes), 3))



    PI = np.zeros(len(centers),dtype=float) 
    # eventually move to classes?
    for n in range(0,len(centers)):
        # get nodes for volume
        pts = get_points(G,centers[n],pos) 
        # calculate volume
        vol = convex_hull_volume_bis(pts)  
        # calculate pressure
        PI[n] = -const.press_alpha*(vol-const.v_0) 



    for node in G.nodes(): 
        # update force on each node  
        force = [0.0,0.0,0.0]

        #Elastic forces due to the cytoskeleton 
        for neighbor in G.neighbors(node):
            a = pos[node]
            b = pos[neighbor]
            
            dist = euclidean_distance(a,b)
            direction = unit_vector(a,b)
            
            magnitude = elastic_force(dist, G[node][neighbor]['l_rest'], mu_apical) 
            force = np.sum([force,magnitude*np.array(direction)],axis=0)
            
            # Force due to myosin
            magnitude = myo_beta*G[node][neighbor]['myosin']
            force = np.sum([force, magnitude*np.array(direction)],axis=0)

        forces[node] = np.add(forces[node], force) 
    
    for center in centers:
        index = np.argwhere(centers==center)[0,0]
        
        pts = circum_sorted[index]
        centroid = np.array([pos[center], pos[center+basal_offset]])
        centroid = np.average(centroid,axis=0)
        
        # pressure for: 
        # apical nodes     
        for i in range(0,len(circum_sorted[index])):
            area, extra = be_area([center,pts[i],pts[i-1]],[center,pts[i],pts[i-1]],pos) 
            magnitude = PI[index]*area[0]*(1/3)
            
            direction = area[1]/np.linalg.norm(area[1]) 
            force = magnitude*direction
            forces[center] = np.add(forces[center],force)
            forces[pts[i-1]] = np.add(forces[pts[i-1]],force)
            forces[pts[i]] = np.add(forces[pts[i]],force)
   
        # pressure for: 
        # basal nodes
            area, extra = be_area([center+basal_offset,pts[i-1]+basal_offset,pts[i]+basal_offset],[center+basal_offset,pts[i-1]+basal_offset,pts[i]+basal_offset],pos) 
            magnitude = PI[index]*area[0]*(1/3)
            direction = area[1]/np.linalg.norm(area[1]) 
            force = magnitude*direction
            forces[center+basal_offset] = np.add(forces[center+basal_offset],force)
            forces[pts[i-1]+basal_offset] = np.add(forces[pts[i-1]+basal_offset],force)
            forces[pts[i]+basal_offset] = np.add(forces[pts[i]+basal_offset],force)

    # pressure for side panels
    # loop through each cell
    for index in range(0,len(circum_sorted)):
        cell_nodes = circum_sorted[index]
        centroid = np.array([pos[centers[index]], pos[centers[index]+basal_offset]])
        centroid = np.average(centroid, axis=0)
        # loop through the 6 faces (or 5 or 7 after intercalation)
        for i in range(0, len(cell_nodes)):
            pts_id = np.array([cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+basal_offset, cell_nodes[i-1]+basal_offset])
            pts_pos = np.array([pos[pts_id[ii]] for ii in range(0,4)])
            # on each face, calculate the center
            center = np.average(pts_pos,axis=0)
            # loop through the 4 triangles that make the face
            for ii in range(0,4):
                pos_side = [center, pts_pos[ii-1], pts_pos[ii]] 
                area = area_side(pos_side) 
                magnitude = PI[index]*area[0]*(1/2)
                
                direction = area[1]/np.linalg.norm(area[1]) 
                force = magnitude*direction
                forces[pts_id[ii-1]] = np.add(forces[pts_id[ii-1]],force)
                forces[pts_id[ii]] = np.add(forces[pts_id[ii]],force)
    
    # Implement bending energy
    # Loop through all alpha, beta pairs of triangles
    for pair in triangles:
        alpha, beta = pair[0], pair[1]
        
        # Apical faces, calculate areas and cross-products 
        A_alpha, A_beta = be_area(alpha, beta, pos)
         
        for node in alpha:
            inda = np.argwhere(alpha==node)[0,0]
            nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
            if node in beta:
                indb = np.argwhere(beta==node)[0,0]
                nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                frce = const.c_ab*bending_energy(nbhrs_alpha, nbhrs_beta, A_alpha, A_beta, pos)
            else:
                frce = const.c_ab*bending_energy(nbhrs_alpha, False, A_alpha, A_beta, pos)
		
            forces[node] = np.add(forces[node],frce)

        for node in beta:
            # don't double count the shared nodes
            indb = np.argwhere(beta==node)[0,0]
            nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
            if node not in alpha:
                frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
            else:			
                frce = const.c_ab*np.array([0.,0.,0.])
            
            forces[node] = np.add(forces[node],frce)

        # Basal faces
        alpha = [alpha[0]+basal_offset, alpha[1]+basal_offset, alpha[2]+basal_offset] 
        beta = [beta[0]+basal_offset, beta[1]+basal_offset, beta[2]+basal_offset] 

        A_alpha, A_beta = be_area(alpha, beta, pos)
        
        for node in alpha:
            inda = np.argwhere(alpha==node)[0,0]
            nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
            if node in beta:
                indb = np.argwhere(beta==node)[0,0]
                nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                frce = const.c_ab*bending_energy(nbhrs_alpha, nbhrs_beta, A_alpha, A_beta, pos)
            else:
                frce = const.c_ab*bending_energy(nbhrs_alpha, False, A_alpha, A_beta, pos)
		
            forces[node] = np.add(forces[node],frce)

        for node in beta:
            # don't double count the shared nodes
            indb = np.argwhere(beta==node)[0,0]
            nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
            if node not in alpha:
                frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
            else:			
                frce = np.array([0.,0.,0.])
            
            forces[node] = np.add(forces[node],frce)


    
    return forces

    

