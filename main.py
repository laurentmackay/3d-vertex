import os

import numpy as np
from  VertexTissue.Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder


from ResearchTools.Geometry import euclidean_distance
from ResearchTools.Dict import  last_dict_value
from ResearchTools.Caching import   cache_file


from VertexTissue.SG import arcs_pit_and_intercalation
from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.funcs_orig import clinton_timestepper
from VertexTissue.globals import pit_centers, inter_edges_middle, inter_edges_middle_bis, inter_edges_outer, inter_edges_outer_bis, inner_arc, outer_arc, myo_beta, l_apical, press_alpha
from VertexTissue.util import arc_to_edges, edge_index, find_first, get_myosin_free_cell_edges, inside_arc
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import SLS_nonlin, crumple, fluid_element, edge_crumpler, extension_remodeller, shrink_edges






try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = './data/'



def get_inter_edges(intercalations=0, outer=False, double=False):
    
    middle_edges = inter_edges_middle[:intercalations] if intercalations<=12 else inter_edges_middle_bis[:intercalations] 
    outer_edges = inter_edges_outer[:intercalations] if intercalations<=12 else inter_edges_outer_bis[:intercalations]

    inter_edges = middle_edges


    if outer:
        if double:
            inter_edges.extend( outer_edges )
        else:
            inter_edges = outer_edges

    return inter_edges


def run(phi0, remodel=True, intercalations=0, outer=False, double=False, viewable=viewable,
        pit_strength=300, clinton_timestepping=False, dt_min=5e-2, basal=False, scale_pit=True, ec=0.2,
        extend=False, contract=True, T1=True, no_pit_T1s=False, SLS=False, SLS_no_extend=False, SLS_no_contract=False,
        fastvol=False, t_final=4e4):
    
    if (contract==False and extend==False) or (SLS_no_contract and SLS_no_extend):
        return
      
    pattern = cache_file(cache_dir=base_path)
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
   
    belt = get_outer_belt(G_apical)

    # p10=pit_strength
    # p03=pit_strength
    # m=(p10-p03)/0.7
    # b=p03-0.3*m
    # pit_strength=m*phi0+b



    inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

    
    if SLS is False:
        k_eff = (phi0-ec)/(1-ec)
    else:
        k_eff=phi0
        
#     if k_eff<=0.01:
#            return
    
    alpha=1
    if scale_pit:
        sigma = (alpha*ec*l_apical*(-1+phi0)+(ec-phi0)*pit_strength*myo_beta)/((-1+ec)*myo_beta)
    else:
        sigma = pit_strength
    # sigma=pit_strength
    t_start = 375


    squeeze = arcs_pit_and_intercalation(G, belt, 
                                           t_1=t_start, 
                                           inter_edges=inter_edges, 
                                           t_intercalate=t_start, 
                                           pit_strength=sigma, 
                                           intercalation_strength=1000, 
                                           basal_intercalation=basal)


    
    kw={'rest_length_func': fluid_element(phi0=phi0, ec=ec, extend=extend, contract=contract) if not SLS else None}

    if remodel:
        kw={**{'maxwell':True, 'maxwell_nonlin': extension_remodeller() }, **kw}

    done=False
    def terminate(*args):
        nonlocal done
        done=True


    def contract_maxwell_branch(i,j,locals=None,**kw):
           G=locals['G']
           basal_offset=G.graph['basal_offset']
           G[i][j]['l_rest']=0.0
           G[i+basal_offset][j+basal_offset]['l_rest']=0.0


    
    if no_pit_T1s: #Do not allow T1s to occur inside the pit
           centers = G.graph['centers']
           circum_sorted=G.graph['circum_sorted']
           pit_indices = [np.argwhere(centers == c)[0,0] for c in pit_centers]
           extra_arcs = [circum_sorted[i] for i in pit_indices]
           inner_bois = arc_to_edges( *extra_arcs)
           for n in np.unique(np.array(inner_bois)[:]):
                for nn in list(G.neighbors(n)):
                      center_ind = np.argwhere(centers==nn)
                      cond = len(center_ind) != 0 and not (nn in pit_centers)
                      if cond:
                                extra_arcs.append(circum_sorted[center_ind[0,0]])
    else:
           extra_arcs=[]
           
    blacklist=arc_to_edges(belt, inner_arc, outer_arc, *extra_arcs)
    
    # b
    #create integrator

#     def extension_remodelling(ell, L, L0):
#         eps=0.0
#         if L>0:
#                 eps=(ell-L0)/L0

#         if (L<=0 and ell>0) or (eps>ec) or (ell<L):
#                 val=(ell-L)

#                 return val
#         else:
#                 return 0.0
  

        
    if (SLS and (SLS_no_contract or SLS_no_extend or ec!=0.0)):
        maxwell_nonlin = SLS_nonlin(ec=ec, contract=not SLS_no_contract, extend=not SLS_no_extend)
    else:
        maxwell_nonlin=None

              
          
#     viewable=False
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, append_to_blacklist=True, RK=4,
                                    intercalation_callback=contract_maxwell_branch if SLS else None,
                                    angle_tol=.01, length_rel_tol=0.05, SLS = False if not SLS else phi0 ,
                                    maxwell_nonlin= maxwell_nonlin,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None } if viewable else False, minimal=False, T1=T1, fastvol=fastvol, **kw)


    print(f'effective spring constant: {k_eff}')



    integrate(5, t_final,
              pre_callback=squeeze,
              dt_init= 1e-3,
              adaptive=True,
              dt_min=max(dt_min*k_eff, 0.1*dt_min),
              adaptation_rate=0.1,
              save_rate=100, 
              view_rate=5,   
              verbose=True,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False)

intercalations=[0, 1, 4, 6, 8, 10, 12, 14 ,16, 18]


if __name__ == '__main__':
     run(0.1, ec=0.1, t_final=2e4, remodel=False,   viewable=True, dt_min=5e-3,  scale_pit=True, no_pit_T1s=True, SLS=True, SLS_no_contract=True, fastvol=True)


