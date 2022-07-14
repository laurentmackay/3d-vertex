import numpy as np
from numba import jit

from VertexTissue.globals import l_apical
from VertexTissue.funcs import euclidean_distance

def edge_crumpler(G, phi0=1.0, ec=0.2):

    def inner(ell,e):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

    return inner

def crumple(phi0=1.0, ec=0.2):

    @jit(nopython=True, cache=True)
    def inner(ell,L0):

        l_rest=L0.copy()
        inds = (ell<(1-ec)*L0).reshape(-1,)
        l_rest[inds] = (1-phi0)*ell[inds]/(1-ec)+ phi0*L0[inds]
        return l_rest

    return inner

def shrink_edges(G, L0_min=None, basal=True):
    basal_offset = G.graph['basal_offset']
    dynamic = L0_min is None
    
    def inner(node,neighbour, L0_min=L0_min):
        
            if dynamic:
                a=G.nodes[node]['pos']
                b=G.nodes[neighbour]['pos']
                L0_min = euclidean_distance(a,b)

            G[node][neighbour]['l_rest'] = L0_min
            
            if basal:
                    if dynamic:
                        a=G.nodes[node+basal_offset]['pos']
                        b=G.nodes[neighbour+basal_offset]['pos']
                        L0_min = euclidean_distance(a,b)
                    G[node+basal_offset][neighbour+basal_offset]['l_rest'] = 0

    return inner


def extension_remodeller(ec=.3):
    
    @jit(nopython=True, cache=True)
    def inner(ells,L0):
        out = np.zeros(L0.shape)
        for i in range(len(L0)):
            L=L0[i,0]
            ell=ells[i,0]
            eps=0
            if L>0:
                    eps=(ell-L)/L
            
            if (L<=0.0 and ell>0.0) or (eps>ec and L<3.4):
                out[i]=(ell-L)-ec*max(L,0)
            else:
                out[i] =0

        return out


    return inner