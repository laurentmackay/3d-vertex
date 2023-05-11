import numpy as np
import VertexTissue.globals as const
from VertexTissue.Geometry import euclidean_distance, triangle_area_vector, triangle_areas_and_vectors, unit_vector_and_dist
import numba
from numba import jit

myo_beta = const.myo_beta 

@jit(nopython=True, cache=True)
def compute_distances(pos, edges):
    N_edges = len(edges)
    dists = np.zeros((N_edges,),dtype=numba.float64)
    
    for i, e in enumerate(edges):
        
        dist = euclidean_distance(pos[e[0]],pos[e[1]])
        dists[i] = dist

    return dists

@jit(nopython=True, cache=True)
def compute_distances_and_directions(pos, edges, ndim=3):
    N_edges = len(edges)
    dists = np.zeros((N_edges,),dtype=numba.float64)
    drx = np.zeros((N_edges, ndim) ,dtype=numba.float64)
    
    
    for i, e in enumerate(edges):
        
        direction, dist = unit_vector_and_dist(pos[e[0]][:ndim],pos[e[1]][:ndim])
        dists[i] = dist
        drx[i] = direction

    return dists, drx

@jit(nopython=True, cache=True, inline='always')
def compute_rod_forces(forces, l_rest, dists, drx, myosin, edges,  mu_apical, ndim=3):

    for i, e in enumerate(edges):
        magnitude = mu_apical*(dists[i] - l_rest[i])
        
        magnitude2 = myo_beta * myosin[i]
    
        force=(magnitude + magnitude2)*drx[i,:ndim]
        forces[e[0]]+=force
        forces[e[1]]-=force

@jit(nopython=True, cache=True, inline='never')
def compute_SLS_forces(forces, l1, l2, dists, drx, myosin, edges,  mu_apical, alpha, ndim=3):
    beta=1-alpha
    for i, e in enumerate(edges):
        magnitude = mu_apical*(dists[i] - alpha*l1[i] - beta*l2[i])
        
        magnitude2 = myo_beta * myosin[i]
    
        force=(magnitude + magnitude2)*drx[i,:ndim]
        forces[e[0]]+=force
        forces[e[1]]-=force
        
@jit(nopython=True, cache=True)
def compute_spring_forces(forces, l_rest, dists, drx, edges,  mu_apical, ndim=3):

    for i, e in enumerate(edges):
        magnitude = (dists[i] - l_rest[i])
            
        force=magnitude*drx[i,:ndim]
        forces[e[0]]+=force
        forces[e[1]]-=force

@jit(nopython=True, cache=True)
def compute_myosin_forces(forces, myosin, drx, edges, ndim=3):

    for i, e in enumerate(edges):
        magnitude = myo_beta*myosin[i]
            
        force=magnitude*drx[i,:ndim]
        forces[e[0]]+=force
        forces[e[1]]-=force

@jit(nopython=True, cache=True, inline='always')
def apply_pressure_and_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds):
    # Implement bending energy
    
    #precompute areas
    pos_side = pos[triangles_sorted.ravel()].reshape(len(triangles_sorted),3,3)
    # pos_side = np.array([pos[tri] for tri in triangles_sorted])
    areas, area_vecs = triangle_areas_and_vectors(pos_side)

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


        bending_forces = const.c_ab * compute_bending_forces(bools[0], bools[1], A_alpha_vec, A_alpha , A_beta_vec, A_beta, delta_alpha, delta_beta)
        
        for node, force in zip(nodes, bending_forces):
            forces[node] += force
inl = 'never'
@jit(nopython=True, cache=True, inline=inl)
def apply_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds):
    # Implement bending energy
    
    #precompute areas
    pos_side = pos[triangles_sorted.ravel()].reshape(len(triangles_sorted),3,3)
    # pos_side = np.array([pos[tri] for tri in triangles_sorted])
    areas, area_vecs = triangle_areas_and_vectors(pos_side)

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


        bending_forces = const.c_ab * compute_bending_forces(bools[0], bools[1], A_alpha_vec, A_alpha , A_beta_vec, A_beta, delta_alpha, delta_beta)
        
        for node, force in zip(nodes, bending_forces):
            forces[node] += force

#def pressure_and_bending_forces( triangles_sorted, shared_inds, alpha_inds, beta_inds, ab_pair_tri_inds, side_face_inds):

@jit(nopython=True, cache=True, inline='never')
def apply_pressure_and_bending_forces(forces, triangles_sorted, pos, shared_inds, alpha_inds, beta_inds, ab_pair_tri_inds, side_face_inds, v0=const.v_0):
    # Implement bending energy
    
    #precompute areas
    pos_side = pos[triangles_sorted.ravel()].reshape(len(triangles_sorted),3,3)
    # pos_side = np.array([pos[tri] for tri in triangles_sorted])
    areas, area_vecs = triangle_areas_and_vectors(pos_side)

    
    #compute the volume of each cell
    vols = np.zeros((len(ab_pair_tri_inds)))
    for i, (cell_tri_inds, side_inds) in enumerate(zip(ab_pair_tri_inds, side_face_inds)):
        vol=0
        #compute the volume contained between each apical-basal triangle pair
        for i_a, i_b in cell_tri_inds:

            a, b, c = triangles_sorted[i_a]
            ap, bp, cp  = triangles_sorted[i_b]

            V1 = abs(np.dot(area_vecs[i_b], pos[a]-pos[ap]))
            V2 = abs(np.dot(area_vecs[i_a], pos[cp]-pos[c]))
            V3 = abs(np.dot(np.cross(pos[cp]-pos[b], pos[a]-pos[bp]), pos[bp]-pos[b]))

            vol+=V1+V2+V3

        vol/=6
        vols[i] = vol
        PI=const.press_alpha*(v0-vol)
        for i_a, i_b in cell_tri_inds:
                a, b, c = triangles_sorted[i_a]
                ap, bp, cp  = triangles_sorted[i_b]

                force_apical = PI*area_vecs[i_a]/3.0
                force_basal = -PI*area_vecs[i_b]/3.0

                forces[a] += force_apical
                forces[b] += force_apical
                forces[c] += force_apical

                forces[ap] += force_basal
                forces[bp] += force_basal
                forces[cp] += force_basal


                # for inds, force in zip(faces, face_forces):


        # pressure for side panels
        # loop through each cell
        
    
        for inds in side_inds:
            # et the positions of the vertices making up each side face
            pos_side = pos[inds]

            # on each face, calculate the center
            center = (pos_side[0]+pos_side[1]+pos_side[2]+pos_side[3]) / 4.0

            # loop through the 4 triangles that make the face
            for j in range(4):
                pos_face = np.vstack((center, pos_side[j-1], pos_side[j]))
                area_vec = triangle_area_vector(pos_face) 
                
                force = PI*area_vec/2.0
                forces[inds[j-1]] += force
                forces[inds[j]] += force


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


        bending_forces = const.c_ab * compute_bending_forces(bools[0], bools[1], A_alpha_vec, A_alpha , A_beta_vec, A_beta, delta_alpha, delta_beta)
        
        for node, force in zip(nodes, bending_forces):
            forces[node] += force


e_hat = np.array([[1,0,0], [0,1,0], [0,0,1]])

@jit(nopython=True, cache=True, inline='always')
def compute_bending_forces(nbhrs_alpha, nbhrs_beta, alpha_vec, A_alpha, beta_vec, A_beta, delta_alpha, delta_beta):


    shape=alpha_vec.shape

    if nbhrs_alpha:
        sums0=np.zeros(shape)
        sums4=np.zeros(shape)
        
    if nbhrs_beta:
        sums1=np.zeros(shape)
        sums3=np.zeros(shape)

    sums2=np.sum(alpha_vec*beta_vec,axis=1).reshape(-1,1)

    for k in range(0,3):
        
        cross_beta = np.cross(delta_beta,e_hat[k])

        if nbhrs_alpha:
            cross_alpha = np.cross(delta_alpha,e_hat[k])
            sums0+=beta_vec[:,k].reshape(-1,1)*cross_alpha/2
            sums4+=alpha_vec[:,k].reshape(-1,1)*cross_alpha/2

        if nbhrs_beta:
            cross_beta = np.cross(delta_beta,e_hat[k])
            sums1+=alpha_vec[:,k].reshape(-1,1)*cross_beta/2
            sums3+=beta_vec[:,k].reshape(-1,1)*cross_beta/2

    
    if nbhrs_alpha and nbhrs_beta:
        return (1.0/(A_alpha*A_beta))*(sums0+sums1) \
                + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3 \
                +(A_beta/A_alpha)*sums4)
    elif nbhrs_alpha:
        return (1.0/(A_alpha*A_beta))*(sums0) \
                + (-sums2/(A_alpha*A_beta)**2)*((A_beta/A_alpha)*sums4)
    elif nbhrs_beta:
        return (1.0/(A_alpha*A_beta))*(sums1) \
        + (-sums2/(A_alpha*A_beta)**2)*((A_alpha/A_beta)*sums3)
    
@jit(nopython=True, cache=True, inline='always')
def cell_volumes(pos, ab_pair_face_inds):
    vols = np.zeros((len(ab_pair_face_inds)))
    for i, cell_inds in enumerate(ab_pair_face_inds):
        vol=0
        
        for a,b,c,ap,bp,cp in cell_inds:
            vol+=triangular_polyhedorn_volume(pos[a],pos[b],pos[c],pos[ap],pos[bp],pos[cp])

        vols[i]=vol

    return vols
    
@jit(nopython=True, cache=True, inline='always')
def triangular_polyhedorn_volume(a,b,c,ap,bp,cp):

    V1 = abs(np.dot(np.cross(cp-ap,bp-ap), a-ap))
    V2 = abs(np.dot(np.cross(a-c,b-c), cp-c))
    V3 = abs(np.dot(np.cross(cp-b,a-bp), b-bp))

    return (V1+V2+V3)/6.0

@jit(nopython=True, cache=True)
def apply_pressure_3D(forces, PI, pos, face_inds,  side_face_inds):
    # pressure for apical and basal faces

    for faces, pressure in zip(face_inds, PI):
        for face in faces:
            pos_face = pos[face]
            area_vec = triangle_area_vector(pos_face)

            force = pressure*area_vec/3.0
            forces[face[0]] += force
            forces[face[1]] += force
            forces[face[2]] += force


            # for inds, force in zip(faces, face_forces):


    # pressure for side panels
    # loop through each cell
    
    for side_inds, pressure in zip(side_face_inds, PI):
        for inds in side_inds:
            # et the positions of the vertices making up each side face
            pos_side = pos[inds]

            # on each face, calculate the center
            center = np.sum(pos_side, axis=0)/4.0

            # loop through the 4 triangles that make the face
            for j in range(4):
                pos_face = np.vstack((center, pos_side[j-1], pos_side[j]))
                area_vec = triangle_area_vector(pos_face) 
                
                force = pressure*area_vec/2.0
                forces[inds[j-1]] += force
                forces[inds[j]] += force

