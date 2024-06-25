import os

import numpy as np
from VertexTissue.Events import CreatePeriodicEvent



from VertexTissue.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.globals import inter_edges_middle, inter_edges_outer, inner_arc, outer_arc, inner_arc_discont, outer_arc_discont, l_apical, pit_strength, myo_beta
from VertexTissue.util import arc_to_edges, circumferential_arc, crossing_edges, get_myosin_free_cell_edges, inside_arc
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.Dict import dict_product
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, edge_crumpler, extension_remodeller, shrink_edges


from VertexTissue.Stochastic import edge_reaction_selector, reaction_times
from VertexTissue.SG import arc_activator, edge_activator

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

def run(phi0, remodel=True, period=50, L0_T1=0.0, verbose=False, belt=True, intercalations=0, outer=False, double=False, rep=0.0, discont=False, ndim=3):
    np.random.seed(int(np.round(rep)))
     
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')
    #
    G, G_apical = tissue_3d( hex=7,  basal=(ndim==3))
    

    belt = get_outer_belt(G_apical)

    # p10=pit_strength
    # p03=pit_strength
    # m=(p10-p03)/0.7
    # b=p03-0.3*m
    # pit_strength=m*phi0+b

        

    inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

    ec=.2

    k_eff = (phi0-ec)/(1-ec)
    alpha=1
    sigma = (alpha*ec*l_apical*(-1+phi0)+(ec-phi0)*pit_strength*myo_beta)/((-1+ec)*myo_beta)
    # sigma=pit_strength
    t_start = 0.01 


    # if stochastic:
    inner_arc = circumferential_arc(14, G, continuous=True)
    

    edges = get_myosin_free_cell_edges(G)
    nodes = np.unique(np.array([e for e in edges]).ravel())
            
    excluded_nodes_inner = nodes[[inside_arc(n, inner_arc, G) or not inside_arc(n, outer_arc, G) or (n in belt)  for n in nodes]].ravel()
    select_inner_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_inner, ignore_activated=True)

    excluded_nodes_outer = nodes[[inside_arc(n, outer_arc, G) or (n in belt)for n in nodes]].ravel()
    select_outer_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_outer, ignore_activated=True)


    activate_edge  = SG.edge_activator(G)
    def inner_intercalation_rxn(*args):
            edge = select_inner_edge()
            activate_edge(edge)

    def outer_intercalation_rxn(*args):
            edge = select_outer_edge()
            activate_edge(edge)

    T_final=4e4

    Rxs_inner = tuple((t, inner_intercalation_rxn, f'Inner intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_init=0, T_final=T_final-5e3))
    Rxs_outer = tuple((t, outer_intercalation_rxn, f'Outer intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_init=0,  T_final=T_final-5e3))



    arcs = (inner_arc, outer_arc) if not discont else (inner_arc_discont, outer_arc_discont)

    squeeze = SG.arc_pit_and_intercalation(G, belt, t_1=t_start, arcs=arcs, inter_edges= [], t_intercalate=t_start, pit_strength=sigma)
    squeeze = SG.pit_and_belt(G, belt)
    squeeze = SG.just_pit(G)

    # if stochastic:
    if outer:
            squeeze.extend(Rxs_outer)
            if double:
                    squeeze.extend(Rxs_inner)
    else:
            squeeze.extend(Rxs_inner)

    
    kw={'rest_length_func': crumple(phi0=phi0)}

    if remodel:
        kw={**{'maxwell':True, 'maxwell_nonlin': extension_remodeller() }, **kw}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done


    if discont:
        arcs=(*arcs[0], *arcs[1])
    blacklist=[]

    # b
    #create integrator
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, append_to_blacklist=False, RK=1,
                                    intercalation_callback=shrink_edges(G, L0=L0_T1),
                                    angle_tol=.01, length_rel_tol=0.05,
                                    player=False, viewer={'button_callback':terminate,'nodeLabels':None } if viewable else False, minimal=False, ndim=ndim, **kw)



    integrate(50, T_final,
              pre_callback=squeeze,
              dt_init=1e-3,
              adaptive=True,
              dt_min=1e-6*k_eff,
              save_rate=100,    
              verbose=verbose,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False)




phi0s=list(reversed([ 0.3,  0.6, 1.0]))

L0_T1s=np.linspace(0,3.4,10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])
L0_T1s = L0_T1s[L0_T1s<=1.2]
L0_T1s = [l_apical, ]
# L0_T1s =[0,]
N_reps=2.0
reps=np.linspace(1.0, N_reps, int(N_reps))

intercalations=[4, 6, 8, 180]
intercalations=[8, 24, 180,]

# kws_baseline = dict_product({'intercalations':0, 'remodel':[True,False], 'L0_T1':0.0,'rep':reps})
kws_middle = dict_product({'intercalations':intercalations, 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})
kws_outer = dict_product({'intercalations':intercalations, 'outer':True, 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})
kws_double = dict_product({'intercalations':intercalations, 'outer':True, 'double':True, 'remodel':[True,False], 'L0_T1':L0_T1s, 'rep':reps})

kws_middle_discont = dict_product({'intercalations':intercalations, 'remodel':[True,False], 'L0_T1':L0_T1s, 'discont':True, 'rep':reps})
kws_outer_discont = dict_product({'intercalations':intercalations, 'outer':True, 'remodel':[True,False], 'L0_T1':L0_T1s,'discont':True, 'rep':reps})
kws_double_discont = dict_product({'intercalations':intercalations, 'outer':True, 'double':True, 'remodel':[True,False], 'L0_T1':L0_T1s,'discont':True, 'rep':reps})



kws = [ *kws_middle, *kws_outer, *kws_double]
# kws = kws_baseline
if __name__ == '__main__':
    
    def foo():
        pass
    # sweep(phi0s, run, kw=kws_double, savepath_prefix=base_path, overwrite=False, pre_process=foo)
    run(1.0, L0_T1=l_apical, intercalations=24, outer=True, double=True, verbose=False, ndim=2)

