import copy
import time
import numpy as np
import pytest
import logging


from scipy.spatial import ConvexHull


from VertexTissue.Tissue import tissue_3d
from VertexTissue.TissueForces import TissueForces, compute_network_indices
from VertexTissue.TissueForcesJitted import cell_volumes, compute_distances_and_directions
from VertexTissue.forces_orig import compute_forces_orig
from VertexTissue.funcs_orig import convex_hull_volume_bis
from VertexTissue.util import get_edge_attribute_array, get_edges_array, get_node_attribute_array, get_points

TOL=1E-12
Nreps=5

logger = logging.getLogger(f"__test_forces__")
logger.setLevel(logging.INFO)

@pytest.fixture(scope="module")
def G3D():
    G,_ = tissue_3d( hex=7,  basal=True)

    
    logger.info('Running initial force computation')
    logger.info('This may take some time due to JIT-compilation...')
		
    compute_forces = TissueForces(G, fastvol=True)

    #dummy code to force JIT-compilation


    pos=get_node_attribute_array(G,'pos')
    edges = get_edges_array(G)

    dists, drx  = compute_distances_and_directions(pos, edges)

    l_rest = get_edge_attribute_array(G, 'l_rest')
    myosin = get_edge_attribute_array(G, 'myosin')
    forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
    logger.info('done.')

    return G
    





def test_fastvol_equiv_to_convexhull(G3D, caplog):
    '''Tests if the `fastvol`  keyword uned in a `VertexTissue.vertex_3d.monolayer_integrator`
       computes volumes that are identical to the convexhull algorithm.

       Such a comparison only makes sense for convex-shaped cells.

        This test is performed by randomly jittering a reference state and computing volumes, so it is run `Nreps` times.
    '''
    
    logger.info(f"Checking the equivalence between fastvol and convexhull algorithm for {Nreps} randomized tissues with {len(G3D.graph['centers'])} cells.")
    logger.info(f"Error Tolerance: {TOL}")
    for offset in (1.0,-1.0):
        cell_type=f"{'non-' if offset<0 else ''}convex"
        logger.info(f"Using {cell_type} cells")
        max_err=0
        for _ in range(Nreps):
            G=copy.deepcopy(G3D)
            err=np.abs(fastvol_relative_diff_to_convexhull(G,  center_offset=offset, randomize_offset=True))
            max_err=max(max_err, np.max(err))
            assert max_err<TOL
        logger.info(f"Largest Error ({cell_type}): {max_err}")




def test_relaxed_reference_state(G3D, caplog):
    '''Ensure that the reference state has negligible forces. Covers elastic and viscoelastic cases.'''

    logger.info("Checking that the unpertured state has zero force.")
    logger.info(f"Error Tolerance: {TOL}")

    G=copy.deepcopy(G3D)

    compute_forces = TissueForces(G)
    compute_viscoelastic_forces = TissueForces(G)


    pos=get_node_attribute_array(G,'pos')
    edges = get_edges_array(G)

    dists, drx  = compute_distances_and_directions(pos, edges)

    l_rest = get_edge_attribute_array(G, 'l_rest')
    myosin = get_edge_attribute_array(G, 'myosin')

    forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
    
    viscoelastic_forces = compute_viscoelastic_forces(l_rest, dists, drx, myosin, edges, pos)

    
    assert np.all(np.abs(forces)<TOL)
    logger.info(f"Largest force component (elastic): {np.max(np.abs(forces))}")
    assert np.all(np.abs(viscoelastic_forces)<TOL)
    logger.info(f"Largest force component (viscoelastic): {np.max(np.abs(viscoelastic_forces))}")


def test_new_force_implementation(G3D, caplog):
    '''Compares new force implementation with Clinton's old implementation.

        Node positions are randomly jittered from the reference state to obtain non-zero forces to compare. 
    '''

    logger.info("Checking that the optimized force routines are equivalent to the original implementation.")
    logger.info(f"Using {Nreps} randomized tissues with {len(G3D.graph['centers'])} cells.")
    logger.info(f"Error Tolerance: {TOL}")
    max_err=0
    for _ in range(Nreps):
        G=copy.deepcopy(G3D)

        compute_forces = TissueForces(G)
        for n in G.node:
            G.node[n]['pos']+=np.random.rand(3,)

        pos=get_node_attribute_array(G,'pos')
        edges = get_edges_array(G)

        dists, drx  = compute_distances_and_directions(pos, edges)

        l_rest = get_edge_attribute_array(G, 'l_rest')
        myosin = get_edge_attribute_array(G, 'myosin')

        forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
        forces_orig=compute_forces_orig(G)
        # viscoelastic_forces = compute_viscoelastic_forces(l_rest, dists, drx, myosin, edges, pos)

        max_err=max(max_err, np.max(np.abs(forces-forces_orig)))
        assert np.all(max_err<TOL)
    logger.info(f"Largest component-wise difference: {max_err}")


def test_speedup(G3D):
        
        logger.info("Checking that the optimized force computations are atleast 10x faster than the original implementation.")
        G=copy.deepcopy(G3D)

        compute_forces = TissueForces(G, fastvol=True)
        for n in G.node:
            G.node[n]['pos']+=np.random.rand(3,)

        pos=get_node_attribute_array(G,'pos')
        edges = get_edges_array(G)

        dists, drx  = compute_distances_and_directions(pos, edges)

        l_rest = get_edge_attribute_array(G, 'l_rest')
        myosin = get_edge_attribute_array(G, 'myosin')
        forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
        
        fold=10
        
        logger.info(f"Using {Nreps} trials for the original implementation and {Nreps*fold} trials for the optimized code.")

        t_new = time.perf_counter()
        for _ in range(Nreps*fold):
            forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
        t_new = time.perf_counter()-t_new

        t_old = time.perf_counter()
        for _ in range(Nreps):
            forces_orig=compute_forces_orig(G)
        t_old = time.perf_counter()-t_old

        logger.info(f"Original Execution Time: {1000*t_old/(Nreps)} ms")
        logger.info(f"Updated Execution Time: {1000*t_new/(Nreps*fold)} ms")
        

        assert (t_new/fold)<t_old/10.0
        logger.info(f"Speedup: {t_old/(t_new/fold)}")


        

def fastvol_relative_diff_to_convexhull(G,  center_offset=0.0, randomize_offset=False, non_convex=False):
    '''Helper function for computing relative differences between `fastvol` and the convexhull algorithm.'''

    _, _, _, _, _, _, _, ab_pair_face_inds = compute_network_indices(G)
    centers=G.graph['centers']
    if center_offset !=0.0:
        for c in centers:
            if not randomize_offset:
                G.node[c]['pos'][-1]+=center_offset
            else:
                G.node[c]['pos'][-1]+=np.random.rand()*center_offset

    pos=get_node_attribute_array(G,'pos')

    fast_vols = cell_volumes(pos, ab_pair_face_inds)
    
    convex_vols = np.array([convex_hull_volume_bis(get_points(G, c, pos) ) for c in centers])

    if center_offset<0.0: #compensate for a non-convex shape
        circum_sorted = G.graph['circum_sorted']
        convex_vols -= np.array(
            [ 
            ConvexHull(np.array([G.node[c]['pos'], *[G.node[n]['pos'] for n in circum]])).volume for c, circum in zip(centers, circum_sorted )
            ] 
            )
        
    
    
    return (convex_vols-fast_vols)/convex_vols    

if __name__=='__main__':
    G,_ = tissue_3d( hex=7,  basal=True)
    test_fastvol_nonconvex(G)
