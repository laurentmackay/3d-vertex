import copy
import time
import numpy as np
import pytest
import pytest_print

from scipy.spatial import ConvexHull


from VertexTissue.Tissue import tissue_3d
from VertexTissue.TissueForces import TissueForces, compute_network_indices
from VertexTissue.TissueForcesJitted import cell_volumes, compute_distances_and_directions
from VertexTissue.forces_orig import compute_forces_orig
from VertexTissue.funcs_orig import convex_hull_volume_bis
from VertexTissue.util import get_edge_attribute_array, get_edges_array, get_node_attribute_array, get_points
TOL=1E-12
Nreps=5

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

    if center_offset<0.0:
        circum_sorted = G.graph['circum_sorted']
        convex_vols -= np.array(
            [ 
            ConvexHull(np.array([G.node[c]['pos'], *[G.node[n]['pos'] for n in circum]])).volume for c, circum in zip(centers, circum_sorted )
            ] 
            )
        
    
    
    return (convex_vols-fast_vols)/convex_vols    


@pytest.fixture(scope="module")
def G3D():
    G,_ = tissue_3d( hex=7,  basal=True)
    return G


def test_fastvol_equiv_to_convexhull(G3D, Nreps=5):
    '''Tests if the `fastvol`  keyword uned in a `VertexTissue.vertex_3d.monolayer_integrator`
       computes volumes that are identical to the convexhull algorithm.

       Such a comparison only makes sense for convex-shaped cells.

        This test is performed by randomly jittering a reference state and computing volumes, so it is run `Nreps` times.
    '''
    for _ in range(Nreps):
        G=copy.deepcopy(G3D)
        err=np.abs(fastvol_relative_diff_to_convexhull(G,  center_offset=1.0, randomize_offset=True))
        assert np.all(err<TOL)

def test_fastvol_nonconvex(G3D):
    '''Tests if the `fastvol`  keyword computes volumes that are smaller than the
        convexhull algorithm for non-convex cells.

        This test is performed by randomly jittering a reference state and computing volumes, so it is run `Nreps` times.
    '''
    for _ in range(Nreps):
        G=copy.deepcopy(G3D)
        err=fastvol_relative_diff_to_convexhull(G,  center_offset=-1, randomize_offset=True)
        assert np.all(np.abs(err)<TOL)




def test_relaxed_reference_state(G3D):
    '''Ensure that the reference state has negligible forces. Covers elastic and viscoelastic cases.'''

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
    assert np.all(np.abs(viscoelastic_forces)<TOL)


def test_new_force_implementation(G3D):
    '''Compares new force implementation with Clinton's old implementation.

        Node positions are randomly jittered from the reference state to obtain non-zero forces to compare. 
    '''


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


        assert np.all(np.abs(forces-forces_orig)<TOL)

def test_speedup(G3D, printer):

    
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
        
        t_new = time.perf_counter()
        for _ in range(Nreps*10):
            forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)
        t_new = time.perf_counter()-t_new

        t_old = time.perf_counter()
        for _ in range(Nreps):
            forces_orig=compute_forces_orig(G)
        t_old = time.perf_counter()-t_old

        print(f"Original Execution Time: {t_old/Nreps} s")
        print(f"Updated Execution Time: {t_new/(Nreps*10)} s")
        print(f"Speedup: {t_old/(t_new/10)}")

        assert (t_new/10.0)<t_old/10.0


        


if __name__=='__main__':
    G,_ = tissue_3d( hex=7,  basal=True)
    test_fastvol_nonconvex(G)