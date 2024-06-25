import numpy as np
from numba import jit

from VertexTissue.globals import l_apical
from ResearchTools.Geometry import euclidean_distance
from VertexTissue.util import has_basal

def edge_crumpler(G, phi0=1.0, ec=0.2):

    def inner(ell,e):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

    return inner

def crumple(phi0=1.0, ec=0.2):

    alpha = ((1.0-phi0)/(1.0-ec))
    # @jit(nopython=True, cache=True)
    def inner(ell,L0):

        l_rest=L0.copy()
        inds = ( ell < (1-ec)*L0 )
        l_rest[inds] = alpha*ell[inds] + phi0*L0[inds]
        return l_rest

    return inner

def fluid_element(phi0=1.0, ec=0.2, extend=False, contract=True):

    alpha = ((1.0-phi0)/(1.0-ec))
    beta = (ec*(phi0-2)+phi0)/(1.0-ec)
    # @jit(nopython=True, cache=True)
    def inner(ell,L0):

        l_rest=L0.copy()
        
        if contract:
            inds =  ell < (1-ec)*L0 
            l_rest[inds] = alpha*ell[inds] + phi0*L0[inds]
        
        if extend:
            inds =  ell > (1+ec)*L0 
            l_rest[inds] = alpha*ell[inds] + beta*L0[inds]

        return l_rest

    return inner

def SLS_nonlin(ec=0.0, contract=True, extend=True):
    contract=np.array(contract)
    extend=np.array(extend)

    def inner(ell, L, L0, ec=ec, SLS_contract=contract, SLS_extend=extend):
            eps=np.zeros(L.shape)
            dLdt=np.zeros(L.shape)

            inds = L>0

            eps[inds]=(ell[inds]-L0[inds])/L0[inds]

            inds = np.logical_and( L<=0, ell>0)
            # i1 = np.logical_and(SLS_contract, np.logical_or(np.logical_or(inds ,  eps<-ec),  np.logical_and(ell>L, L0>L)))
            if contract:
                    inds = np.logical_or(np.logical_or(inds ,  eps<-ec),  np.logical_and(ell>L, L0>L))
            # i2 = np.logical_and(SLS_extend, np.logical_or(np.logical_or(inds ,  eps>ec),   np.logical_and(ell<L, L0<L)))
            # inds = np.logical_or(i1,i2)
            if extend:
                    inds = np.logical_or(np.logical_or(inds ,  eps>ec),   np.logical_and(ell<L, L0<L))

            dLdt[inds] = (ell[inds]-L[inds])

            return dLdt
    
    return inner

def shrink_edges(G, L0=None, basal=True):
    basal = basal and has_basal(G)
    
    if has_basal(G):
        basal_offset = G.graph['basal_offset']

    dynamic = L0 is None
    
    def inner(node,neighbour, L0_min=L0, **kw):
        
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
    
    # @jit(nopython=True, cache=True)
    def inner(ells,L0):
        out = np.zeros(L0.shape)
        for i in range(len(L0)):
            L=L0[i]
            ell=ells[i]
            eps=0
            if L>0:
                    eps=(ell-L)/L
            
            if (L<=0.0 and ell>0.0) or (eps>ec and L<3.4):
                out[i]=(ell-L)-ec*max(L,0)
            else:
                out[i] =0

        return out


    return inner