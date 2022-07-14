import numpy as np
from numba.typed import List
import numba

from .funcs import *
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

def TissueForces(G=None, ndim=3, minimal=False, compute_pressure=True):



    # def zero_force_dict(G):
    #     return {node: np.zeros(ndim ,dtype=float) for node in G.nodes()}
#    def zero_force_vec(G):
#         return np.zeros((len(G),ndim) ,dtype=float)
    
    @jit(nopython=True, cache=True)
    def compute_distances_and_directions(pos, edges):
        N_edges = len(edges)
        dists = np.zeros((N_edges,1),dtype=numba.float64)
        drx = np.zeros((N_edges, ndim) ,dtype=numba.float64)
       
        
        for i, ed in enumerate(edges):
            
            direction, dist = unit_vector_and_dist(pos[ed[0]],pos[ed[1]])
            dists[i] = dist
            drx[i] = direction

        return dists, drx

    mu_apical = const.mu_apical  
    press_alpha = const.press_alpha 
    
    @jit(nopython=True, cache=True)
    def compute_rod_forces(forces, l_rest, dists, drx, myosin, edges):

        for i, e in enumerate(edges):
            magnitude = mu_apical*(dists[i] - l_rest[i])
            magnitude2 = myo_beta*myosin[i]
        
            force=(magnitude + magnitude2)*drx[i,:ndim]
            forces[e[0]]+=force
            forces[e[1]]-=force

        



        

    # def compute_tissue_forces_2D(force_dict=None, G=G, compute_distances= True):


    #     circum_sorted = G.graph['circum_sorted']


    #     if force_dict is None:
    #         force_dict = zero_force_dict(G)

    #     forces = compute_rod_forces(force_dict, G=G, compute_distances = compute_distances)


    #     grad = np.zeros((2,))

    #     if compute_pressure:
        
    #         # pre-calculate magnitude of pressure
    #         # index of list corresponds to index of centers list
    #         # eventually move to classes?
    #         for pts in circum_sorted:
    #             coords=np.array([pos[i] for i in pts])
    #             x=coords[:,0]
    #             y=coords[:,1]
    #             area = polygon_area(x,y)
    #             pressure = (area-const.A_0)
    #             for i, pt in enumerate(pts):
    #                 eps=1e-5
    #                 x[i]+=eps/2
    #                 grad[0]=(polygon_area(x,y)-area)/(eps/2)
    #                 x[i]-=eps

    #                 y[i]+=eps
    #                 grad[1]=(polygon_area(x,y)-area)/eps
    #                 y[i]-=eps

    #                 force = -press_alpha*pressure*grad
    #                 forces[pt] += force

    # @jit(nopython=True, cache=True)
    # def bending_forces(forces, triangles, pos):

        

    @jit(nopython=True, cache=True)
    def apply_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds):
        # Implement bending energy
        
        #precompute areas
        pos_side = pos[triangles_sorted.ravel()].reshape(len(triangles_sorted),3,3)
        areas, area_vecs = area_side2(pos_side)

        # Loop through all alpha, beta pairs of triangles
        # apply forces to the nodes that are shared by the pair, only in alpha, or only in beta
        bool_vec = ((True,True), (True,False), (False,True))
        for bools, inds in zip(bool_vec, (shared_inds, alpha_inds, beta_inds)):

            nodes=inds[:,0]
            i_alpha=inds[:,1]
            i_beta=inds[:,2]
            A_alpha = areas[i_alpha].reshape((-1,1))
            A_alpha_vec = np.asfortranarray(area_vecs[i_alpha])
            A_beta = areas[i_beta].reshape((-1,1))
            A_beta_vec = np.asfortranarray(area_vecs[i_beta])
            delta_alpha = pos[inds[:,4]]-pos[inds[:,3]]
            delta_beta =  pos[inds[:,6]]-pos[inds[:,5]]


            bending_forces = const.c_ab * bending_energy_3(bools[0], bools[1], A_alpha_vec, A_alpha , A_beta_vec, A_beta, delta_alpha, delta_beta)
            
            for node, force in zip(nodes, bending_forces):
                forces[node] += force


    @jit(nopython=True, cache=True)
    def apply_pressure(forces, PI, pos, face_inds,  side_face_inds):

        for faces, pressure in zip(face_inds, PI):
            pos_faces = pos[faces.ravel()].reshape(len(faces),3,3)
            area_vecs = area_side_vec2(pos_faces)

            face_forces = pressure*area_vecs/3.0
            for inds, force in zip(faces, face_forces):
                forces[inds[0]] += force
                forces[inds[1]] += force
                forces[inds[2]] += force

        # pressure for side panels
        # loop through each cell
        for side_inds, pressure in zip(side_face_inds, PI):
            
            #get the positions of the vertices making up each face
            pos_side = pos[side_inds.ravel()].reshape(len(side_inds),4,3)

            # on each face, calculate the center
            centers = np.sum(pos_side, axis=1)/4.0
 
            # loop through the faces
            for i, inds in enumerate(side_inds):
                # loop through the 4 triangles that make the face
                for j in range(4):
                    pos_face = np.vstack((centers[i], pos_side[i,j-1], pos_side[i,j]))
                    area_vec = area_side_vec(pos_face) 
                    
                    force = pressure*area_vec/2.0
                    forces[inds[j-1]] += force
                    forces[inds[j]] += force

    

    centers = G.graph['centers']


       

    ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted = compute_network_indices(G) 

    # @profile
    def compute_tissue_forces_3D(l_rest, dists, drx, myosin, edges, pos, recompute_indices=False):
        nonlocal ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted

        if recompute_indices:
            ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted = compute_network_indices(G) 

        forces = np.zeros((len(G),ndim) ,dtype=float)
        compute_rod_forces(forces, l_rest, dists, drx, myosin, edges)


        if compute_pressure:
        
            # pre-calculate magnitude of pressure
            # index of list corresponds to index of centers list
            PI = np.zeros(len(centers),dtype=float) 
            # eventually move to classes?
            for n in range(len(centers)):
                # get nodes for volume
                pts = get_points(G, centers[n], pos) 
                # calculate volume
                vol = convex_hull_volume(pts)  
                # calculate pressure
                DV = vol-const.v_0
                if abs(DV)<1e-7:
                    DV=0
                PI[n] = -press_alpha*DV

            apply_pressure(forces, PI, pos, ab_face_inds, side_face_inds)
        
     
        # bending_forces(forces, triangles, pos)
        apply_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds)


        return forces



    if minimal:
         compute_forces=compute_rod_forces
    elif ndim == 3 and not minimal:
        compute_forces=compute_tissue_forces_3D
    elif ndim == 2:
        compute_forces=compute_tissue_forces_2D

    compute_forces.compute_distances_and_directions = compute_distances_and_directions

    return compute_forces

def compute_network_indices(G):
    triangles = G.graph['triangles']
    def bending_indices():

        
        shared_inds = np.zeros((len(triangles),2,7),dtype=int)
        alpha_inds = np.zeros((len(triangles),7),dtype=int)
        beta_inds = np.zeros((len(triangles),7),dtype=int)

        triangles_sorted = np.unique(triangles.reshape(len(triangles)*2,3), axis=0)
        # G.graph['triangles_sorted'] = triangles_sorted

        for i in range(len(triangles)):
            alpha, beta = triangles[i]

            i_alpha = np.argwhere(np.all(alpha == triangles_sorted, axis = 1))[0][0]
            i_beta  = np.argwhere(np.all( beta == triangles_sorted, axis = 1))[0][0]

            # Apical faces, calculate areas and cross-products z
            # A_alpha, A_alpha_vec, A_beta, A_beta_vec = be_area_2(pos[alpha], pos[beta])
            shared=0
            for inda in range(3):
                node = alpha[inda]
                nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
                if node in beta:
                    indb = np.argwhere(beta==node)[0][0]
                    nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                    shared_inds[i,shared,:] = (node, i_alpha, i_beta, *nbhrs_alpha, *nbhrs_beta)
                    shared+=1
                    
                else:
                    alpha_inds[i,:] = (node, i_alpha, i_beta, *nbhrs_alpha, *nbhrs_alpha)
                    

            for indb in range(3):
                # don't double count the shared nodes
                node = beta[indb]

                if node not in alpha:
                    nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                    beta_inds[i,:] = (node, i_alpha, i_beta, *nbhrs_beta, *nbhrs_beta)

        shared_inds=shared_inds.reshape((len(shared_inds)*2,7))
        return shared_inds, alpha_inds, beta_inds, triangles_sorted

    circum_sorted = G.graph['circum_sorted']
    centers = G.graph['centers']
    basal_offset = G.graph['basal_offset']
        #pressure for apical and basal faces

    ab_face_inds=[]
    for center, pts in zip(centers, circum_sorted):  
        face_inds = []
        for i in range(len(pts)):
            a=(center, pts[i], pts[i-1])
            b=(center+basal_offset, pts[i-1]+basal_offset, pts[i]+basal_offset)
            face_inds.extend((a,b))


        ab_face_inds.append(np.array(face_inds))


    ab_face_inds=List(ab_face_inds)
    side_face_inds = []

    for cell_nodes in circum_sorted:
            face_inds=[[cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+basal_offset, cell_nodes[i-1]+basal_offset] for i in range(len(cell_nodes))]
            side_face_inds.append(np.array(face_inds))
    side_face_inds=List(side_face_inds)


    shared_inds, alpha_inds, beta_inds, triangles_sorted = bending_indices()

    return  ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted
