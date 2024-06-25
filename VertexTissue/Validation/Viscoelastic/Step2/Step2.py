import os

import numpy as np



from VertexTissue.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.globals import inter_edges_middle, inter_edges_outer, inner_arc, outer_arc, pit_strength, myo_beta, l_apical
from VertexTissue.Dict import dict_product
from VertexTissue.Memoization import  function_call_savepath
from VertexTissue.util import arc_to_edges
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, edge_crumpler, extension_remodeller, shrink_edges






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



def run(phi0, remodel=True, cable=True, L0_T1=0.0, verbose=False, belt=True, intercalations=0, outer=False, double=False, viewable=viewable):
    
    
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    

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
    t_start = 1 
    squeeze = SG.arc_pit_and_intercalation(G, belt, t_1=t_start,
                                         inter_edges=inter_edges, t_intercalate=t_start, pit_strength=sigma)



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



    blacklist=arc_to_edges(belt, inner_arc, outer_arc)

    # b
    #create integrator
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, RK=1,
                                    intercalation_callback=shrink_edges(G, L0=L0_T1),
                                    angle_tol=.01, length_rel_tol=0.05,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None} if viewable else False, minimal=False, **kw)



    integrate(5, 4e4,
              pre_callback=squeeze,
              dt_init=1e-3,
              adaptive=True,
              dt_min=5e-1*k_eff,
              save_rate=100,    
              verbose=verbose,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False)



def main():
    run(750,  phi0=0.3, cable=True)

phi0s=np.array(list(reversed([ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])))

L0_T1s=np.linspace(0,3.4,10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])
L0_T1s = L0_T1s[L0_T1s<=1.2]
# L0_T1s=np.array([0.0,])

kws_baseline = dict_product({'intercalations':0, 'L0_T1':0.0, 'remodel':True})
kws_middle = dict_product({'intercalations':[4,6, 18], 'remodel':[True,False], 'L0_T1':L0_T1s})
kws_outer = dict_product({'intercalations':[4, 6, 24], 'outer':True, 'remodel':[True,False], 'L0_T1':L0_T1s})
kws_double = dict_product({'intercalations':[4, 6], 'outer':True,'double':True, 'remodel':[True,False], 'L0_T1':L0_T1s})

# kws = [ *kws_middle, *kws_outer, *kws_double]
kws = kws_baseline
if __name__ == '__main__':
    
    def foo(*args):
        pass
    # sweep(phi0s, run, kw=kws, savepath_prefix=base_path, overwrite=False, pre_process=foo)
    run(.3, L0_T1=0,  outer=True, verbose=True, viewable=True)

