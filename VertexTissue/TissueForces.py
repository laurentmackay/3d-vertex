import numpy as np
from numba.typed import List
import numba
from numba import jit

from VertexTissue.TissueForcesJitted import elastic_forces_jitted, viscoelastic_forces_jitted
from VertexTissue.funcs_orig import convex_hull_volume_bis
from .TissueForcesJitted import apply_bending_forces, apply_pressure_3D, cell_volumes, compute_distances_and_directions, compute_myosin_forces, compute_rod_forces, compute_spring_forces, handle_pressure_3D_fast

from ResearchTools.Geometry import euclidean_distance, unit_vector_and_dist, triangle_areas_and_vectors, triangle_area_vector
# from .Geometry  import convex_hull_volume as convex_hull_volume_bis
from . import globals as const
from .globals import press_alpha
from .util import conditional_decorator, get_edge_attribute_array, get_edges_array, get_node_attribute_array, get_points, has_basal, polygon_area

# l_apical = const.l_apical 
# l_depth = const.l_depth 

# l0_apical = l_apical
# l0_basal = l_apical 
# l0_wall = l_depth 


mu_basal = const.mu_basal          
mu_wall = const.mu_wall          

eta = const.eta 



def spring_forces(G, rest_length_func=None, ndim=3):
        '''Convenience function for computing the forces due to the edges
        
            Use TissueForces for a more optimized experience
        '''

        if rest_length_func is not None:
            def get_rest_lengths(ell,L0):
                    return rest_length_func(ell,L0)
        else:
            def get_rest_lengths(ell,L0):
                    return L0

        pos=get_node_attribute_array(G,'pos')

        edges = get_edges_array(G)

        dists, drx  = compute_distances_and_directions(pos, edges)
        L0 = get_edge_attribute_array(G, 'l_rest')

        forces = np.zeros((len(G),ndim) ,dtype=float)

        l_rest = get_rest_lengths(dists, L0)

        compute_spring_forces(forces, l_rest, dists, drx, edges, const.mu_apical, ndim=ndim)

        return forces

def myosin_forces(G, ndim=3):
        '''Convenience function for computing the forces due to myosin on the edges
        
            Use TissueForces for a more optimized experience
        '''

        pos=get_node_attribute_array(G,'pos')

        edges = get_edges_array(G)

        _, drx  = compute_distances_and_directions(pos, edges)
        myosin = get_edge_attribute_array(G, 'myosin')

        forces = np.zeros((len(G),ndim) ,dtype=float)

        # l_rest = get_rest_lengths(dists, L0)

        compute_myosin_forces(forces, myosin, drx, edges, ndim=ndim)

        return forces

def bending_forces(G, triangulation=None, ndim=3):
    '''Convenience function for computing the bending forces
    
        Use TissueForces for a more optimized experience
    '''

    pos=get_node_attribute_array(G,'pos')

    forces = np.zeros((len(G),ndim) ,dtype=float)
    
    if triangulation is None:    
        _, _, shared_inds, alpha_inds, beta_inds, triangles_sorted, _= compute_network_indices(G) 
    else:
        shared_inds, alpha_inds, beta_inds, triangles_sorted = triangulation



    apply_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds)


    return forces

#@profile
def pressure(G, pos, centers, v0=const.v_0, fastvol=False, ab_pair_face_inds=None):
    if fastvol:
        vols = cell_volumes(pos, ab_pair_face_inds)
    else:
        vols = np.array([convex_hull_volume_bis(get_points(G, c, pos) ) for c in centers])
    
    return const.press_alpha*(v0-vols)

def pressure_forces(G, faces=None, ndim=3, v0=None):
    '''Convenience function for computing the pressure forces
    
        Use TissueForces for a more optimized experience
    '''

    pos=get_node_attribute_array(G,'pos')



    forces = np.zeros((len(G),ndim) ,dtype=float)
    if ndim==3:
        centers = G.graph['centers']
        if faces is None:
            ab_face_inds, side_face_inds, _, _, _, _, _ = compute_network_indices(G) 
        else:
            ab_face_inds, side_face_inds = faces

        if v0 is None:
            v0=const.v_0

        apply_pressure_forces_3D(forces, G, pos, ab_face_inds, side_face_inds, centers, v0)
    else:
        if v0 is None:
           v0=const.A_0

        circum_sorted= G.graph['circum_sorted']
        handle_pressure_2D(forces, pos, circum_sorted, v0)


    return forces


#@profile
# @jit(nopython=True, cache=True, inline='never')
def apply_pressure_forces_3D(forces ,G, pos, ab_face_inds, side_face_inds, centers, v0=const.v_0, fastvol=False, ab_pair_face_inds=None):
    PI=pressure(G, pos, centers, v0=v0, fastvol=fastvol, ab_pair_face_inds=ab_pair_face_inds)
    #PI = np.zeros(centers.shape)

    apply_pressure_3D(forces, PI, pos, ab_face_inds, side_face_inds)


# def handle_pressure_2D(forces, pos, circum_sorted, v0=const.A_0):

#          for pts in circum_sorted:
#                 coords=np.array([pos[i] for i in pts])

#                 x=coords[:,0]
#                 y=coords[:,1]

#                 area = polygon_area(x,y)
#                 pressure = (area-v0)

#                 for i, pt in enumerate(pts):
#                     eps=1e-5
#                     x[i]+=eps
#                     grad[0]=(polygon_area(x,y)-area)/(eps)
#                     x[i]-=eps

#                     y[i]+=eps
#                     grad[1]=(polygon_area(x,y)-area)/eps
#                     y[i]-=eps

#                     force = -press_alpha*const.l_depth*pressure*grad
#                     forces[pt] += force

def TissueForces(G=None, ndim=3, minimal=False, compute_pressure=True, SLS=False, fastvol=False):


    press_alpha = const.press_alpha 
    mu_apical = const.mu_apical
    myo_beta = const.myo_beta
    centers = G.graph['centers']


        

    grad = np.zeros((2,))
    def handle_pressure_2D(forces, pos, ab_face_inds, side_face_inds, ab_pair_face_inds, v0=const.A_0):

         for pts in circum_sorted:
                coords=np.array([pos[i] for i in pts])

                x=coords[:,0]
                y=coords[:,1]

                area = polygon_area(x,y)
                pressure = (area-v0)

                for i, pt in enumerate(pts):
                    eps=1e-5
                    x[i]+=eps
                    grad[0]=(polygon_area(x,y)-area)/(eps)
                    x[i]-=eps

                    y[i]+=eps
                    grad[1]=(polygon_area(x,y)-area)/eps
                    y[i]-=eps

                    force = -press_alpha*const.l_depth*pressure*grad
                    forces[pt] += force
    #@profile
    
    def handle_pressure_3D(forces, pos, ab_face_inds, side_face_inds, ab_pair_face_inds, v0=const.v_0):      
            apply_pressure_forces_3D(forces, G, pos, ab_face_inds, side_face_inds,  centers, v0=v0, ab_pair_face_inds=ab_pair_face_inds, fastvol=fastvol)


                
                



    
    def compute(pos, l_rest, dists, drx, myosin, edges,  side_face_inds, ab_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, ab_pair_face_inds, v0=const.v_0):
        if SLS is False:
            forces = elastic_forces_jitted(pos, l_rest, dists, drx, myosin, edges,  side_face_inds, ab_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, ab_pair_face_inds, SLS, ndim, compute_pressure, fastvol, mu_apical, myo_beta, press_alpha, v0=v0)
        else:
            forces = viscoelastic_forces_jitted(pos, l_rest, dists, drx, myosin, edges,  side_face_inds, ab_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, ab_pair_face_inds, SLS, ndim, compute_pressure, fastvol, mu_apical, myo_beta, press_alpha, v0=v0)

             
        if compute_pressure and not fastvol:
            handle_pressure(forces, pos, ab_face_inds, side_face_inds, ab_pair_face_inds, v0=v0)


        return forces
    
    # if fastvol:
    #     compute = compute_jitted


        
   
    
    ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, circum_sorted, ab_pair_face_inds = compute_network_indices(G) 

    # @jit(nopython=True, cache=True, inline='never')
    def compute_tissue_forces(l_rest, dists, drx, myosin, edges, pos, recompute_indices=False, v0=const.v_0):
        nonlocal ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, circum_sorted, ab_pair_face_inds

        if recompute_indices:
            ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, circum_sorted, ab_pair_face_inds = compute_network_indices(G) 

        return compute(pos, l_rest, dists, drx, myosin, edges,  side_face_inds, ab_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, ab_pair_face_inds, v0=v0)


    # #
    
    # #@jit(nopython=True, cache=True, inline='always')
    # def compute(pos, l_rest, dists, drx, myosin, edges, side_face_inds,  shared_inds, alpha_inds, beta_inds, triangles_sorted, ab_pair_tri_inds, v0=None):
    #     forces = np.zeros(pos.shape ,dtype=float)
    #     if SLS==1.0:
    #         compute_rod_forces(forces, l_rest, dists, drx, myosin, edges, const.mu_apical, ndim=ndim)
    #     else:
    #         compute_SLS_forces(forces, l_rest[0], l_rest[1], dists, drx, myosin, edges, const.mu_apical, SLS, ndim=ndim)

    #     if compute_pressure:
    #         handle_pressure_3D_fast(forces, pos, ab_face_inds, side_face_inds,   ab_pair_face_inds, v0=v0)

    #     if ndim==3:
    #         apply_bending_forces(forces, triangles_sorted,  pos, shared_inds, alpha_inds, beta_inds, ab_pair_tri_inds, side_face_inds, v0=v0)

    #     return forces

    if minimal:
         compute_forces=compute_rod_forces
    else:
        compute_forces=compute_tissue_forces
        if ndim==3:
            handle_pressure = handle_pressure_3D_fast if fastvol else handle_pressure_3D
        elif ndim == 2:
            handle_pressure = handle_pressure_2D


    return compute_forces

def compute_network_indices(G):
    is_basal = has_basal(G)

    circum_sorted = G.graph['circum_sorted']
    centers = G.graph['centers']

    if is_basal:
        basal_offset = G.graph['basal_offset']

    def bending_indices():

        triangles = G.graph['triangles']
        if is_basal:
            triangles = np.vstack((triangles, triangles+basal_offset))

        shared_inds = np.zeros((len(triangles),2,7),dtype=int)
        alpha_inds = np.zeros((len(triangles),7),dtype=int)
        beta_inds = np.zeros((len(triangles),7),dtype=int)
        
        triangles_sorted = np.unique(triangles.reshape(len(triangles)*2,3), axis=0)
        # G.graph['triangles_sorted'] = triangles_sorted

        for i in range(len(triangles)):
            # for offset in (0, basal_offset):
            alpha, beta = triangles[i]
            # alpha+=offset
            # beta+=offset
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
        # shared_inds = np.vstack((shared_inds[:,0,:], shared_inds[:,1,:]))
        return shared_inds, alpha_inds, beta_inds, triangles_sorted


    shared_inds, alpha_inds, beta_inds, triangles_sorted = bending_indices()

    #pressure for apical and basal faces

    ab_face_inds=[]
    ab_pair_face_inds=[]
    # ab_pair_tri_inds=[]
    for center, pts in zip(centers, circum_sorted):  
        face_inds = []
        face_pair_inds = []
        # tri_pair_inds = []
        for i in range(len(pts)):
            a=(center, pts[i], pts[i-1])
            # i_a = np.argwhere(np.all( a == triangles_sorted, axis = 1))[0][0]
            if is_basal:
                b=(center+basal_offset, pts[i-1]+basal_offset, pts[i]+basal_offset)
                face_inds.extend((a,b))
                face_pair_inds.append((*a,*(center+basal_offset, pts[i]+basal_offset, pts[i-1]+basal_offset)))
                # i_b = np.argwhere(np.all( (b[0],b[2],b[1]) == triangles_sorted, axis = 1))[0][0]
                # tri_pair_inds.append((i_a,i_b))
            else:
                face_inds.append(a)


        ab_face_inds.append(np.array(face_inds))
        if is_basal:
             ab_pair_face_inds.append(np.array(face_pair_inds))
            #  ab_pair_tri_inds.append(np.array(tri_pair_inds))


    ab_face_inds=List(ab_face_inds)
    ab_pair_face_inds=List(ab_pair_face_inds)
    # ab_pair_tri_inds=List(ab_pair_tri_inds)

    ab_pair_face_inds._make_immutable()
    # ab_pair_tri_inds._make_immutable()

    side_face_inds = []
    if is_basal:
        for cell_nodes in circum_sorted:
                face_inds=[[cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+basal_offset, cell_nodes[i-1]+basal_offset] for i in range(len(cell_nodes))]
                side_face_inds.append(np.array(face_inds))

    side_face_inds=List(side_face_inds)


    

    return  ab_face_inds, side_face_inds, shared_inds, alpha_inds, beta_inds, triangles_sorted, circum_sorted, ab_pair_face_inds#, ab_pair_tri_inds



# # principal unit vectors e_x, e_y, e_z
# e_hat = np.array([[1,0,0], [0,1,0], [0,0,1]])

# @jit(nopython=True, cache=True)
# def bending_energy(nbhrs_alpha, nbhrs_beta, alpha_vec, A_alpha, beta_vec, A_beta, delta_alpha, delta_beta):


#     shape=alpha_vec.shape

#     if nbhrs_alpha:
#         sums0=np.zeros(shape)
#         sums4=np.zeros(shape)
        
#     if nbhrs_beta:
#         sums1=np.zeros(shape)
#         sums3=np.zeros(shape)

#     sums2=np.sum(alpha_vec*beta_vec,axis=1).reshape(-1,1)

#     for k in range(0,3):
        
#         cross_beta = np.cross(delta_beta,e_hat[k])

#         if nbhrs_alpha:
#             cross_alpha = np.cross(delta_alpha,e_hat[k])
#             sums0+=beta_vec[:,k].reshape(-1,1)*cross_alpha/2
#             sums4+=alpha_vec[:,k].reshape(-1,1)*cross_alpha/2

#         if nbhrs_beta:
#             cross_beta = np.cross(delta_beta,e_hat[k])
#             sums1+=alpha_vec[:,k].reshape(-1,1)*cross_beta/2
#             sums3+=beta_vec[:,k].reshape(-1,1)*cross_beta/2

    
#     if nbhrs_alpha and nbhrs_beta:
#         return (1.0/(A_alpha*A_beta))*(sums0+sums1) \
#                 + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3 \
#                 +(A_beta/A_alpha)*sums4)
#     elif nbhrs_alpha:
#         return (1.0/(A_alpha*A_beta))*(sums0) \
#                 + (-sums2/(A_alpha*A_beta)**2)*((A_beta/A_alpha)*sums4)
#     elif nbhrs_beta:
#         return (1.0/(A_alpha*A_beta))*(sums1) \
#         + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3)

# @jit(nopython=True, cache=True)
# def apply_pressure_3D(forces, PI, pos, face_inds,  side_face_inds):
#     # pressure for apical and basal faces

#     for faces, pressure in zip(face_inds, PI):
#         for face in faces:
#             pos_face = pos[face]
#             area_vec = triangle_area_vector(pos_face)

#             force = pressure*area_vec/3.0
#             forces[face[0]] += force
#             forces[face[1]] += force
#             forces[face[2]] += force


#             # for inds, force in zip(faces, face_forces):


#     # pressure for side panels
#     # loop through each cell
    
#     for side_inds, pressure in zip(side_face_inds, PI):
#         for inds in side_inds:
#             # et the positions of the vertices making up each side face
#             pos_side = pos[inds]

#             # on each face, calculate the center
#             center = np.sum(pos_side, axis=0)/4.0

#             # loop through the 4 triangles that make the face
#             for j in range(4):
#                 pos_face = np.vstack((center, pos_side[j-1], pos_side[j]))
#                 area_vec = triangle_area_vector(pos_face) 
                
#                 force = pressure*area_vec/2.0
#                 forces[inds[j-1]] += force
                # forces[inds[j]] += force


# def bending_energy(nbhrs_alpha, nbhrs_beta, A_alpha, A_beta, pos):
    
#     # principal unit vectors e_x, e_y, e_z
#     e = np.array([[1,0,0], [0,1,0], [0,0,1]])
    
#     # initialize the sums to zero
#     sums = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

#     for k in range(0,3):
#         # sum (1) and (5) use the alpha cell
#         if nbhrs_alpha != False:
#             cross = np.cross(np.asarray(pos[nbhrs_alpha[-1]])-np.asarray(pos[nbhrs_alpha[0]]),e[k])
#             sums[0] += A_beta[1][k]*(1/2)*cross
#             sums[4] += A_alpha[1][k]*(1/2)*cross

#         # sum (2) and (4) use the beta cell
#         if nbhrs_beta != False:
#             cross = np.cross(np.asarray(pos[nbhrs_beta[-1]])-np.asarray(pos[nbhrs_beta[0]]),e[k])
#             sums[1] += A_alpha[1][k]*(1/2)*cross
#             sums[3] += A_beta[1][k]*(1/2)*cross

#         # sum (3)
#         sums[2] += A_alpha[1][k]*A_beta[1][k]

#     return np.array((1/(A_alpha[0]*A_beta[0]))*(sums[0]+sums[1]) \
#             + (-sums[2]/(A_alpha[0]*A_beta[0])**2)*((A_alpha[0]/A_beta[0])*sums[3] \
#             +(A_beta[0]/A_alpha[0])*sums[4]))



# #@profile
# @jit(nopython=True, cache=True)
# def bending_energy_2(nbhrs_alpha, nbhrs_beta, alpha_vec, A_alpha, beta_vec, A_beta, pos_alpha_A, pos_alpha_B, pos_beta_A, pos_beta_B):

#     # sums = np.zeros((5,3),dtype=numba.float32)
#     # sums = np.zeros((5,3),dtype=float)#numba.float32)
#     delta_alpha = pos_alpha_B-pos_alpha_A
#     delta_beta = pos_beta_B-pos_beta_A
#     cross_alpha = np.cross(delta_alpha,e_hat)
#     cross_beta = np.cross(delta_beta,e_hat)
#     # for k in range(0,3):
#     #     # sum (1) and (5) use the alpha cell
        
#     #     if nbhrs_alpha != False:
            
#     #         sums[0] += beta_vec[k]*(1/2)*cross_alpha[k]
#     #         sums[4] += alpha_vec[k]*(1/2)*cross_alpha[k]

#     #     # sum (2) and (4) use the beta cell
#     #     if nbhrs_beta != False:
#     #         sums[1] += alpha_vec[k]*(1/2)*cross_beta[k]
#     #         sums[3] += beta_vec[k]*(1/2)*cross_beta[k]

#     #     # sum (3)
#     #     sums[2] += alpha_vec[k]*beta_vec[k]

#     sums2=np.dot(alpha_vec,beta_vec)

#     if nbhrs_alpha:
#         sums0=np.dot(beta_vec,cross_alpha)/2
#         sums4=np.dot(alpha_vec,cross_alpha)/2

#     if nbhrs_beta:
#         sums1=np.dot(alpha_vec,cross_beta)/2
#         sums3=np.dot(beta_vec,cross_beta)/2
    
#     if nbhrs_alpha and nbhrs_beta:
#         return (1.0/(A_alpha*A_beta))*(sums0+sums1) \
#                 + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3 \
#                 +(A_beta/A_alpha)*sums4)
#     elif nbhrs_alpha:
#         return (1.0/(A_alpha*A_beta))*(sums0) \
#                 + (-sums2/(A_alpha*A_beta)**2)*((A_beta/A_alpha)*sums4)
#     elif nbhrs_beta:
#         return (1.0/(A_alpha*A_beta))*(sums1) \
#         + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3)

