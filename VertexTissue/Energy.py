import numpy as np
from VertexTissue.TissueForces import compute_network_indices

from VertexTissue.util import get_node_attribute_array, get_edge_attribute_array, get_edges_array, get_points, has_basal
from VertexTissue.TissueForcesJitted import compute_distances
from VertexTissue.funcs_orig import convex_hull_volume_bis
from .Geometry  import convex_hull_volume 
from VertexTissue.Geometry import triangle_areas_and_vectors

import VertexTissue.globals as const

# @profile
def get_cell_volumes_bis(G, pos, centers):
    vols = np.zeros(centers.shape)
    for i, c in enumerate(centers):
        pts = get_points(G, c, pos)
        vols[i] = convex_hull_volume_bis(pts)  
    return vols

# @profile
def get_cell_volumes(G, pos, centers):
    vols = np.zeros(centers.shape)
    for i, c in enumerate(centers):
        pts = get_points(G, c, pos)
        vols[i] = convex_hull_volume(pts)  
    return vols

# @profile
def network_energy(G, phi0=1.0, ec=0.2, triangulation=None, get_volumes=get_cell_volumes_bis, bending=True, spring=True, pressure=True, v0=const.v_0, press_alpha=const.press_alpha):
    
    pos = get_node_attribute_array(G,'pos')
    
    centers=G.graph['centers']
    
    
    E=0

    if pressure:
        vols=get_volumes(G, pos, centers)
        pressure_energy = press_alpha*np.sum((vols-v0)**2)
        E+=pressure_energy

    
    if spring:
        edges = get_edges_array(G)

        ell = compute_distances(pos, edges)
        L0 =  get_edge_attribute_array(G,'l_rest')
        
        Uspring = deformation_energies(ell,L0, phi0=phi0,ec=ec)
        spring_energy = np.sum(Uspring)
        E+=spring_energy
   


    if bending:
        if triangulation is None:
            _, _, _, _, triangle_inds, triangles_sorted, _ = compute_network_indices(G) 
        else:
            triangle_inds, triangles_sorted = triangulation

        pos_side = pos[triangles_sorted.ravel()].reshape(len(triangles_sorted),3,3)
        areas, area_vecs = triangle_areas_and_vectors(pos_side)

        A_hat = area_vecs/areas.reshape(-1,1)

        areas, area_vecs = triangle_areas_and_vectors(pos_side)


        A_hat_alpha =A_hat[triangle_inds[:,1]]
        A_hat_beta = A_hat[triangle_inds[:,2]]
        bending_energy = const.c_ab*np.sum((1-np.sum(A_hat_alpha*A_hat_beta, axis=1))**2)
        E+=bending_energy


    return E



def deformation_energies(ell,L0,phi0=1.0,ec=0.2):
    L0_tilde = (1-ec)*L0
    crumpling = ell<L0_tilde

    U=np.zeros(ell.shape)
    U[~crumpling] = (const.mu_apical/2)*(ell[~crumpling]-L0[~crumpling])**2
    U[crumpling] = (const.mu_apical/(2*(1-ec)))*(phi0*(ell[crumpling]-L0_tilde[crumpling])**2 + ec*(L0[crumpling]*L0_tilde[crumpling]-ell[crumpling]**2))

    return U
