import os

import numpy as np



from VertexTissue.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.globals import inter_edges_middle, inter_edges_outer, inner_arc, outer_arc
from VertexTissue.util import arc_to_edges, get_myosin_free_cell_edges, inside_arc
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.Dict import dict_product
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, edge_crumpler, extension_remodeller, shrink_edges


from VertexTissue.Stochastic import edge_reaction_selector, reaction_times
from VertexTissue.SG import edge_activator

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

def extending_edge_length(G, edge = None):
        b=G.node[edge[0]]['pos']
        c=G.node[edge[1]]['pos']

        return euclidean_distance(b,c)

def get_inter_edges(intercalations=0, outer=False, double=False):
    inter_edges=inter_edges_middle[:intercalations]

    if outer:
        if double:
            inter_edges.extend(inter_edges_outer[:intercalations])
        else:
            inter_edges = inter_edges_outer[:intercalations]

    return inter_edges

def run(phi0, remodel=True, cable=True, L0_T1=0.0, verbose=False, belt=True, intercalations=0, outer=False, double=False, rep=0.0):
    
   
    
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    edges = get_myosin_free_cell_edges(G)
    nodes = np.unique(np.array([e for e in edges]).ravel())

    if belt:
        belt = get_outer_belt(G_apical)


        
    excluded_nodes_inner = nodes[[inside_arc(n, inner_arc, G) or not inside_arc(n, outer_arc, G) or (n in belt) or  (n in outer_arc) or (n in inner_arc) for n in nodes]].ravel()
    select_inner_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_inner)

    excluded_nodes_outer = nodes[[inside_arc(n, outer_arc, G) or (n in belt) or (n in outer_arc) for n in nodes]].ravel()
    select_outer_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_outer)


    activate_edge  = edge_activator(G)
    def inner_intercalation_rxn(*args):
        edge = select_inner_edge()
        activate_edge(edge)

    def outer_intercalation_rxn(*args):
        edge = select_outer_edge()
        activate_edge(edge)

    T_final=4e4

    Rxs_inner = tuple((t, inner_intercalation_rxn, f'Inner intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_final=T_final-2e4))
    Rxs_outer = tuple((t, outer_intercalation_rxn, f'Outer intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_final=T_final-2e4))

    belt = get_outer_belt(G_apical)

    p10=5
    p03=5
    m = (p10 - p03)/0.7
    b = p03 - 0.3*m
    pit_strength = m*phi0 + b



    t_start = 1 
    squeeze = SG.arc_pit_and_intercalation(G, belt, t_1=t_start,
                                         inter_edges=[], t_intercalate=t_start, pit_strength=pit_strength)

    if outer:
        squeeze.extend(Rxs_outer)
        if double:
            squeeze.extend(Rxs_inner)
    else:
        squeeze.extend(Rxs_inner)


    kw = {'rest_length_func': crumple(phi0=phi0)}

    if remodel:
        kw={**{'maxwell':True, 'maxwell_nonlin': extension_remodeller() }, **kw}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done


    blacklist=arc_to_edges(belt, inner_arc, outer_arc)

    #create integrator
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, RK=2,
                                    intercalation_callback=shrink_edges(G, L0=L0_T1),
                                    angle_tol=.01, length_rel_tol=0.05,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None} if viewable else False, minimal=False, **kw)



    integrate(5, T_final,
              pre_callback=squeeze,
              dt_init=1e-3,
              adaptive=True,
              dt_min=5e-2,
              save_rate=100,
              verbose=verbose,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False)




phi0s=list(reversed([0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

L0_T1s=np.linspace(0,3.4,10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])
L0_T1s = L0_T1s[L0_T1s<=1.2]
# L0_T1s =[0,]
N_reps=3
reps=np.linspace(1,N_reps,N_reps)
# kws_baseline = dict_product({'intercalations':0, 'remodel':[True,False], 'L0_T1':0.0,'rep':reps})
kws_middle = dict_product({'intercalations':[4,6, 8], 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})
kws_outer = dict_product({'intercalations':[4, 6, 8], 'outer':True, 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})
kws_double = dict_product({'intercalations':[4, 6, 8], 'outer':True, 'double':True, 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})

kws = [ *kws_middle, *kws_outer, *kws_double]
# kws = kws_baseline
if __name__ == '__main__':
    
    def foo():
        pass
    sweep(phi0s, run, kw=kws, savepath_prefix=base_path, overwrite= False, pre_process=foo)
    # run(0.3, L0_T1=0, intercalations=20, outer=True, double=True,verbose=True)

