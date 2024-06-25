import os

import numpy as np
from .Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder
from VertexTissue.Energy import network_energy
from VertexTissue.Stochastic import edge_reaction_selector, reaction_times



from ResearchTools.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from ResearchTools.Geometry import euclidean_distance, unit_vector
from VertexTissue.funcs_orig import clinton_timestepper, convex_hull_volume_bis, get_points
import VertexTissue.globals as const
from VertexTissue.globals import inter_edges_middle, inter_edges_middle_bis, inter_edges_outer, inter_edges_outer_bis, inner_arc, outer_arc, pit_strength, myo_beta, l_apical, press_alpha
from ResearchTools.Dict import dict_product, dict_product_nd_shape, last_dict_value
from ResearchTools.Caching import  cached, cache_file, keyword_vals
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
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

def extending_edge_length(G, edge = None):
        b=G.node[edge[0]]['pos']
        c=G.node[edge[1]]['pos']

        return euclidean_distance(b,c)

_, G_apical = tissue_3d( hex=7,  basal=True)
belt = get_outer_belt(G_apical)
half_belt = belt[:int(len(belt)/2+1)]

def lumen_depth(G):


        z0 = np.mean([G.node[n]['pos'][-1]  for n in G.neighbors(0)])

        return np.mean([ G.node[n]['pos'][-1]-z0 for n in belt])

def invagination_depth(G):
        basal_offset = G.graph['basal_offset']

        z0 = np.mean([G.node[n]['pos'][-1]  for n in G.neighbors(basal_offset)])

        return np.mean([ G.node[n+basal_offset]['pos'][-1]-z0 for n in belt])

# def invagination_depth_inner(G):
#         basal_offset = G.graph['basal_offset']

#         z0 = np.mean([G.node[n]['pos'][-1]  for n in G.neighbors(0)])

#         return np.mean([ G.node[n+basal_offset]['pos'][-1]-z0 for n in belt])

def final_lumen_depth(d):
        return lumen_depth(last_dict_value(d))

def final_depth(d):
        return invagination_depth(last_dict_value(d))

# def final_depth_inner(d):
#         return invagination_depth_inner(last_dict_value(d))

def depth_timeline(d):
        return np.array([(t, invagination_depth(d[t])) for t in d])

def belt_width(G):
        belt_pos = np.array([ G.node[n]['pos'][:2] for n in belt])
        return np.mean([ max([ euclidean_distance(p,pp) for pp in belt_pos]) for p in belt_pos])

def final_width(d):
        return belt_width(last_dict_value(d))

def arc_width(G, arc=belt):
        arc_pos = np.array([ G.node[n]['pos'][:2] for n in arc])
        return np.mean([ max([ euclidean_distance(p,pp) for pp in arc_pos]) for p in arc_pos])

def final_outer_arc_width(d):
        return arc_width(last_dict_value(d), arc=outer_arc)

def final_inner_arc_width(d):
        return arc_width(last_dict_value(d), arc=inner_arc)

def final_arc_ratio(d, arc='inner'):
        w1=final_width(d)

        if arc=='inner':
               w2=final_inner_arc_width(d)
        elif arc=='outer':
                w2=final_outer_arc_width(d)

        return w1/w2

def width_timeline(d):
        return np.array([(t, belt_width(d[t])) for t in d])


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
def extension(G, intercalations=0, outer=False, double=False, **kw):
        inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

        lens = [euclidean_distance(G.nodes[e[0]]['pos'], G.nodes[e[1]]['pos']) for e in inter_edges]
        if lens:
                return max(lens)
        else:
                return 0

def final_extension(d, **kw):
        return extension(last_dict_value(d), **kw)

def extension_timeline(d, **kw):
        return np.array([(t, extension(G, **kw)) for t, G in d.items()])

def cone_slope(G, outer=True, double=False,**kw):
       r1=belt_width(G)/2
       r2=arc_width(G, arc=outer_arc)/2
       dz=inter_arc_distance(G, outer=outer, double=double)

       return (r1-r2)/(2*dz)

def final_cone_slope(d,**kw):
       return cone_slope(last_dict_value(d), **kw)
       


def inter_arc_distance(G, outer=False, double=False, **kw):
        if outer:
                arc_1=belt
        else:
                arc_1=outer_arc

        if outer and not double:
                arc_2=outer_arc
        else:
                arc_2=inner_arc

        z_1=np.mean([G.node[n]['pos'][-1]  for n in arc_1])
        z_2=np.mean([G.node[n]['pos'][-1]  for n in arc_2])

        return z_1-z_2
        
def final_inter_arc_distance(d,**kw):
        return inter_arc_distance(last_dict_value(d),**kw)

def inter_arc_distance_timeline(d,**kw):
        return np.array([(t, inter_arc_distance(G, **kw)) for t, G in d.items()])

def final_inter_arc_depth(d):
        return inter_arc_distance(last_dict_value(d),outer=True,double=True)

def angle(G, intercalations=0, outer=False, double=False, **kw):
        inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

        angs = [  buckle_angle_finder(G, edge=e)(G) for e in inter_edges]

        if not angs:
                return np.nan
        else:
                return max(angs)

def extension(G, intercalations=0, outer=False, double=False, basal=False, summary=max, **kw):
        inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)
        if basal:
                inter_edges = np.array(inter_edges) + G.graph['basal_offset']
        angs = [  buckle_angle_finder(G, edge=e, basal=basal)(G) for e in inter_edges]
        lens = np.array([  euclidean_distance(G.node[e[0]]['pos'], G.node[e[1]]['pos']) for e in inter_edges])

        if not angs:
                return np.nan
        else:
                return summary(np.cos(angs)*lens)

def final_angle(d, **kw):
    # kw['intercalations']=min(kw['intercalations'],6)
    return angle(last_dict_value(d), **kw)  

def angle_timeseries(d, **kw):
    return np.array([(t, angle(G, **kw)) for t, G in d.items()])

def extension_timeseries(d, tmin=0, **kw):
    return np.array([(t, extension(G, **kw)) for t, G in d.items() if t>=tmin])

def final_corner_angle(d, **kw):
    kw['intercalations']=min(kw['intercalations'],6)
    return angle(last_dict_value(d), **kw)  

def run(phi0, remodel=True, cable=True, L0_T1=0.0, verbose=False, belt=True, intercalations=0, 
        outer=False, double=False, viewable=viewable, stochastic=False, press_alpha=press_alpha,
        pit_strength=300, clinton_timestepping=False, dt_min=5e-2, basal=False, scale_pit=True, mu_apical=const.mu_apical, ec=0.2,
        extend=False, contract=True, T1=True, edge_ratio=0, no_pit_T1s=False, SLS=False, SLS_no_extend=False, SLS_no_contract=False,
        constant_pressure_intercalations=False, fastvol=False, fluid_spokes=False, t_final=4e4):
    
    if (contract==False and extend==False) or (SLS_no_contract and SLS_no_extend):
        return
    
    if t_final>4e4:
          t0=4e4
          v=keyword_vals()
          rez=cached(cache_dir=base_path, t_final=4e4)
          print(rez)
    else:
          t0=0
    
    pattern = cache_file(cache_dir=base_path)
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
   
    belt = get_outer_belt(G_apical)

    # p10=pit_strength
    # p03=pit_strength
    # m=(p10-p03)/0.7
    # b=p03-0.3*m
    # pit_strength=m*phi0+b

    const.press_alpha=press_alpha
    const.mu_apical=mu_apical

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

    if clinton_timestepping:
        clinton_timestep, uncontracted = clinton_timestepper(G, inter_edges)

    if stochastic:

        edges = get_myosin_free_cell_edges(G)
        nodes = np.unique(np.array([e for e in edges]).ravel())
               
        excluded_nodes_inner = nodes[[inside_arc(n, inner_arc, G) or not inside_arc(n, outer_arc, G) or (n in belt) or  (n in outer_arc) or (n in inner_arc) for n in nodes]].ravel()
        select_inner_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_inner)

        excluded_nodes_outer = nodes[[inside_arc(n, outer_arc, G) or (n in belt) or (n in outer_arc) for n in nodes]].ravel()
        select_outer_edge = edge_reaction_selector(G, excluded_nodes=excluded_nodes_outer)


        activate_edge  = SG.edge_activator(G)
        def inner_intercalation_rxn(*args):
                edge = select_inner_edge()
                activate_edge(edge)

        def outer_intercalation_rxn(*args):
                edge = select_outer_edge()
                activate_edge(edge)

        

        Rxs_inner = tuple((t, inner_intercalation_rxn, f'Inner intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_final=t_final-2e4))
        Rxs_outer = tuple((t, outer_intercalation_rxn, f'Outer intercalation triggered at t={t}')for t in reaction_times(n=intercalations, T_final=t_final-2e4))




    squeeze = SG.arc_pit_and_intercalation(G, belt, 
                                           t_1=t_start, 
                                           inter_edges=inter_edges if not stochastic else [], 
                                           t_intercalate=t_start, 
                                           pit_strength=sigma, 
                                           intercalation_strength=1000, 
                                           basal_intercalation=basal, 
                                           edge_ratio=edge_ratio)

    if stochastic:
        if outer:
                squeeze.extend(Rxs_outer)
                if double:
                        squeeze.extend(Rxs_inner)
        else:
                squeeze.extend(Rxs_inner)

    
    kw={'rest_length_func': fluid_element(phi0=phi0, ec=ec, extend=extend, contract=contract) if not SLS else None}

    if remodel:
        kw={**{'maxwell':True, 'maxwell_nonlin': extension_remodeller() }, **kw}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    def label_contracted(i,j, **kw):
        inds = edge_index(G, inter_edges)
        k = find_first( edge_index(G, (i,j)) == inds )
        uncontracted[ k ] = False

    def contract_maxwell(i,j,locals=None,**kw):

           G=locals['G']
           basal_offset=G.graph['basal_offset']
           G[i][j]['l_rest']=0.0
           G[i+basal_offset][j+basal_offset]['l_rest']=0.0

    N_cells=len(G.graph['centers'])
    v0=np.ones((N_cells,))*const.v_0
    def constant_pressure(i,j,locals=None):
        G = locals['G']
        if clinton_timestepping:
                inds = edge_index(G, inter_edges)
                k = find_first( edge_index(G, (i,j)) == inds )
                uncontracted[ k ] = False

        
        centers = locals['centers']
        PI = locals['PI']
        pos = locals['pos']
        four_cells = list({ k for k in G.neighbors(i) if k in centers}.union( { k for k in G.neighbors(j) if k in centers}))
        inds = [np.argwhere(centers == c)[0,0] for c in four_cells]
        curr_vols = [convex_hull_volume_bis(get_points(G, c, pos) ) for c in four_cells]
        v0[inds]=curr_vols+PI[inds]/const.press_alpha
        print(i,j)


    
    if no_pit_T1s:
           centers = G.graph['centers']
           circum_sorted=G.graph['circum_sorted']
           pit_indices = [np.argwhere(centers == c)[0,0] for c in const.pit_centers]
           extra_arcs = [circum_sorted[i] for i in pit_indices]
           inner_bois = arc_to_edges( *extra_arcs)
           for n in np.unique(np.array(inner_bois)[:]):
                for nn in list(G.neighbors(n)):
                      center_ind = np.argwhere(centers==nn)
                      cond = len(center_ind) != 0 and not (nn in const.pit_centers)
                      if cond:
                                extra_arcs.append(circum_sorted[center_ind[0,0]])
        #    inner_bois = arc_to_edges( *extra_arcs)
        #    for ab in inner_bois:
        #         G[ab[0]][ab[1]]['wtv']=1.0
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

    if (SLS and fluid_spokes):
        centers  = [*G.graph['centers']]
        for a,b in G.edges():
              if (a in centers) or (b in centers):
                    G[a][b]['ec']=0.0
                    G[a][b]['SLS_contract']=True
                    G[a][b]['SLS_extend']=True
              else:
                    G[a][b]['ec']=ec
                    G[a][b]['SLS_contract']=not SLS_no_contract
                    G[a][b]['SLS_extend']=not SLS_no_extend
              
          
#     viewable=False
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, append_to_blacklist=True, RK=4,
                                    intercalation_callback=label_contracted if clinton_timestepping else (contract_maxwell if SLS else None),
                                    angle_tol=.01, length_rel_tol=0.05, SLS = False if not SLS else phi0 ,
                                    maxwell_nonlin= maxwell_nonlin,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None } if viewable else False, minimal=False, T1=T1, fastvol=fastvol, **kw)


    print(f'effective spring constant: {k_eff}')



    integrate(5, t_final, t=t0,
              pre_callback=squeeze,
              dt_init= 0.5 if clinton_timestepping else 1e-3,
              adaptive=True,
              timestep_func=clinton_timestep if clinton_timestepping else None,
              dt_min=max(dt_min*k_eff, 0.1*dt_min),
              adaptation_rate=1 if clinton_timestepping else 0.1,
              save_rate=100, 
              view_rate=5,   
              verbose=True,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False)

intercalations=[0, 1, 4, 6, 8, 10, 12, 14 ,16, 18]

def main():
    run(750,  phi0=0.3, cable=True)

phi0s = np.array(list(reversed([ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, 1.0])))
phi0_SLS = np.array(list(reversed([0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, 1.0])))
phi0_SLS = np.array(list(reversed([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, 1.0])))
# phi0s = np.array(list(reversed([ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, 1.0])))

L0_T1s=np.linspace(0, l_apical, 10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])
L0_T1s = L0_T1s[L0_T1s<=1.2]
# L0_T1s=np.array([0.0,])
L0_T1s = [L0_T1s[0], (L0_T1s[0]+ L0_T1s[-1])/2, L0_T1s[-1]]
L0_T1s = [ (L0_T1s[-1])/2, L0_T1s[-1], 3*(L0_T1s[-1])/2, 4*(L0_T1s[-1])/2, l_apical]
# L0_T1s = [0.0,]
L0_T1s=l_apical

remodel=False
kws_baseline = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False}
kws_middle = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s}
kws_outer = {'intercalations':intercalations, 'outer':True, 'remodel':remodel, 'L0_T1':L0_T1s}
kws_double = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':L0_T1s}

kws_baseline_soft = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'mu_apical':const.mu_apical/4}

ecs = np.array([0.0, *np.linspace(0.05, 0.25, 10)])
kws_baseline_thresh = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs}
kws_baseline_thresh_no_scale = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False}
kws_baseline_thresh_no_scale_no_T1 = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'T1':False}
kws_baseline_thresh_no_scale_no_T1_edge = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'T1':False, 'edge_ratio':1.0}

kws_baseline_thresh_no_scale_slice = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True}
kws_inter_thresh_no_scale = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True}
kws_baseline_thresh_no_scale_slice_ext = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True, 'extend':True, 'contract':False}
kws_inter_thresh_no_scale_ext = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True ,'extend':True,'contract':False}
kws_baseline_thresh_no_scale_slice_sym = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True, 'extend':True}
kws_inter_thresh_no_scale_sym = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True ,'extend':True}

kws_inter_thresh_no_scale_mid={'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False, 'ec':0.1, 'scale_pit':False, 'no_pit_T1s':True , 'extend':[True,False], 'contract':[True,False]}



kws_SLS_baseline_thresh = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'ec':ecs,'fastvol':True}
kws_SLS_baseline_thresh_ext = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_contract':True, 'ec':ecs,'fastvol':True}
kws_SLS_baseline_thresh_con = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':True, 'ec':ecs,'fastvol':True}

kws_SLS_baseline_thresh_all = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_contract':[True, False], 'SLS_no_extend':[True, False], 'ec':ecs, 'fastvol': True}


kws_SLS_baseline = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'ec':0.0, 'fastvol': True}
kws_SLS_middle = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'ec':0.0, 'fastvol': True}
kws_SLS_outer = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'outer':True, 'no_pit_T1s':True, 'SLS':True, 'ec':0.0, 'fastvol': True}

kws_SLS_middle_all = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':[True, False], 'SLS_no_contract':[True, False], 'ec':0.0, 'fastvol': True}
kws_SLS_outer_all = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':[True, False], 'SLS_no_contract':[True, False], 'outer':True, 'ec':0.0, 'fastvol': True}

kws_SLS_middle_sym = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'ec':0.0}
kws_SLS_outer_sym = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'outer':True, 'ec':0.0}


kws_SLS_baseline_con = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':True, 'ec':0.0, 'fastvol': True}
kws_SLS_baseline_ext = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_contract':True, 'ec':0.0, 'fastvol': True}


kws_SLS_middle_con = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':True, 'ec':0.0, 'fastvol': True}
kws_SLS_middle_ext = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_contract':True, 'ec':0.0, 'fastvol': True}

kws_SLS_outer_con = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_extend':True,  'outer':True, 'ec':0.0, 'fastvol': True}
kws_SLS_outer_ext = {'intercalations':intercalations, 'L0_T1':L0_T1s, 'remodel':False,  'scale_pit':False, 'no_pit_T1s':True, 'SLS':True, 'SLS_no_contract':True,  'outer':True, 'ec':0.0, 'fastvol': True}


kws_baseline_thresh_extend = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs,'extend':True, 'contract':False}
kws_baseline_thresh_no_scale_extend = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True, 'contract': False}
kws_baseline_thresh_no_scale_no_T1_extend = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True, 'contract': False, 'T1':False}
kws_baseline_thresh_no_scale_no_T1_extend_edge = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True, 'contract': False, 'T1':False, 'edge_ratio':1.0}


kws_baseline_thresh_no_scale_no_T1_all = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':[True, False], 'contract': [False, False], 'T1':False}


kws_baseline_thresh_sym = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs,'extend':True}
kws_baseline_thresh_no_scale_sym = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True}
kws_baseline_thresh_no_scale_no_T1_sym = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True, 'T1':False}
kws_baseline_thresh_no_scale_no_T1_sym_edge = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'ec':ecs, 'scale_pit':False, 'extend':True, 'T1':False, 'edge_ratio':1.0}


kws_baseline_fine = {'intercalations':0, 'L0_T1':L0_T1s, 'remodel':False, 'dt_min':5e-3}
kws_middle_fine = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'dt_min':5e-3}
kws_outer_fine = {'intercalations':intercalations, 'outer':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'dt_min':5e-3}
kws_double_fine = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'dt_min':5e-3}

kws_baseline_smolpit = {'intercalations':0, 'remodel':remodel, 'L0_T1':L0_T1s,'scale_pit':False, 'pit_strength':96.99}
kws_middle_smolpit = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s,'scale_pit':False, 'pit_strength':96.99}
kws_outer_smolpit = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'outer':True,'scale_pit':False, 'pit_strength':96.99}

kws_baseline_no_scale = {'intercalations':0, 'remodel':remodel, 'L0_T1':L0_T1s,'scale_pit':False}
kws_middle_no_scale = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s,'scale_pit':False}
kws_outer_no_scale = {'intercalations':intercalations, 'remodel':remodel, 'outer':True, 'L0_T1':L0_T1s,'scale_pit':False}

kws_middle_basal = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'basal':True}
kws_middle_basal_hi = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'basal':True, 'press_alpha':0.046}
kws_middle_tests = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'press_alpha':[0.046, 0.00735], 'pit_strength':[540,300], 'clinton_timestepping':[True, False]}

kws_middle_clinton = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'clinton_timestepping':True}
kws_middle_clinton_hi_pressure = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'press_alpha':0.046, 'clinton_timestepping':True}
kws_strong_pit_middle_clinton = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True}
kws_middle_hi_pressure = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'press_alpha':0.046}
kws_strong_pit_middle_clinton_hi_pressure = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True, 'press_alpha':0.046 }


kws_strong_pit_baseline = {'intercalations':0, 'L0_T1':0.0, 'remodel':False, 'pit_strength':540}
kws_strong_pit_middle = {'intercalations': intercalations, 'remodel': remodel, 'L0_T1': L0_T1s, 'pit_strength': 540}
kws_strong_pit_middle_hi_pressure = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540,  'press_alpha':0.046 }
kws_strong_pit_outer = {'intercalations':intercalations, 'outer':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540}
kws_strong_pit_double = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540}

# kws_strong_pit_baseline = {'intercalations':0, 'L0_T1':0.0, 'remodel':False, 'pit_strength':540}
kws_strong_pit_middle_clinton = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True}
kws_strong_pit_middle_clinton_hi_pressure = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True, 'press_alpha':0.046 }
kws_strong_pit_outer_clinton = {'intercalations':intercalations, 'outer':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True}
kws_strong_pit_double_clinton = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':L0_T1s, 'pit_strength':540, 'clinton_timestepping':True}

clinton_baseline = {'intercalations':0, 'remodel':False, 'L0_T1':l_apical, 'press_alpha':0.046 }
clinton_middle = {'intercalations':intercalations, 'remodel':False, 'L0_T1':l_apical, 'press_alpha':0.046 }
clinton_outer = {'intercalations':intercalations, 'outer':True, 'remodel':False, 'L0_T1':l_apical, 'press_alpha':0.046 }
clinton_double = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':False, 'L0_T1':l_apical, 'press_alpha':0.046 }

clinton_middle_stochastic = {'intercalations':intercalations, 'remodel':False, 'L0_T1':l_apical, 'stochastic': True  }
clinton_outer_stochastic = {'intercalations':intercalations, 'outer':True, 'remodel':False, 'L0_T1':l_apical, 'stochastic': True  }
clinton_double_stochastic = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':False, 'L0_T1':l_apical, 'stochastic': True }

naught_middle = {'intercalations':intercalations, 'remodel':remodel, 'L0_T1':0 }
naught_outer = {'intercalations':intercalations, 'outer':True, 'remodel':remodel, 'L0_T1':0 }
naught_double = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':0 }

naught_middle_remodel = {'intercalations':intercalations, 'remodel':True, 'L0_T1':0 }
naught_outer_remodel = {'intercalations':intercalations, 'outer':True, 'remodel':True, 'L0_T1':0 }
naught_double_remodel = {'intercalations':intercalations, 'outer':True,'double':True, 'remodel':True, 'L0_T1':0 }

# kws = [*kws_middle, *kws_outer, *kws_double]
kws = kws_double
if __name__ == '__main__':
    
    def foo(*args):
        pass

#     sweep(np.flip(phi0_SLS), run, kw=kws_SLS_outer_all, savepath_prefix=base_path, overwrite=False, pre_process=foo)
    run(0.1, ec=0.1, L0_T1=l_apical, remodel=False,   viewable=True, verbose=True, dt_min=5e-3,  scale_pit=False, no_pit_T1s=True, SLS=True, SLS_no_contract=True, fastvol=True)

