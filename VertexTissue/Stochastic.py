import numpy as np
from ResearchTools.Geometry import unit_vector

from VertexTissue.util import get_myosin_free_cell_edges



def SSA_choose_rx(props, waiting_time=False):

    cumsum = np.cumsum(props)
    prop_tot = cumsum[-1]
    if prop_tot<=0:
        return None
    r=np.random.rand()*prop_tot

    for rx, cumprop in enumerate(cumsum):
        if r<=cumprop:
            break
    if waiting_time:
        tau = np.log(1/np.random.rand())/prop_tot
        return rx, tau
    else:
        return rx

# def reaction_events(rxns, rxn_fun_gen, T_final=None):
#     taus=np.array([tau for  rx, tau in rxns])
#     times=np.cumsum(taus)
#     if T_final is not None:
#         times *= T_final*(1-np.mean(np.diff(taus)))/times[-1]
#     return tuple((t,rxn_fun_gen(rx[0])) for rx, t in zip(rxns, times))

def reaction_times(n=1, T_init=0, T_final=None, pad=True):
    ''' Returns an array of exponentially-spaced times, sutiable for crude stochastic reaction simulations.'''
    taus = [np.log(1/np.random.rand()) for _ in range(n)]
    times=T_init+np.cumsum(taus)
    
    if len(times)==0:
        return []

    if T_final is not None:
        times *= T_final / times[-1]
        if pad:
            times *= (1-np.mean(np.diff(taus)))

    return times


def edge_reaction_selector(G, edges=None, center=0,  excluded_nodes=None, ignore_activated=False):
    if edges is None:
        edges = get_myosin_free_cell_edges(G, excluded_nodes=excluded_nodes)

    def propensities():
        c=G.nodes[center]['pos'][:2]

        a = np.array([G.nodes[e[0]]['pos'][:2] for e in edges])
        b = np.array([G.nodes[e[1]]['pos'][:2] for e in edges])
        d=(a+b)/2
        ab = np.array([unit_vector(aa,bb) for aa,bb in zip(a,b)])
        dc = np.array([unit_vector(dd,c) for dd in d])
        dot = np.sum(dc*ab,axis=-1)
        theta = np.arccos(dot)

        props = (1-np.abs(np.cos(theta)))
        if ignore_activated:
            props[[any([ G[n][e[0]]['myosin']!=0 for n in G.neighbors(e[0])]) for e in edges]]=0
            props[[any([ G[n][e[1]]['myosin']!=0 for n in G.neighbors(e[1])]) for e in edges]]=0
            
        return props

    def select_reaction(waiting_time=False):
        choice = SSA_choose_rx(propensities(), waiting_time=waiting_time)

        if choice is None:
            return None

        if waiting_time:
            return edges[choice[0]], choice[1]
        else:
            return edges[choice]
        


    return select_reaction
