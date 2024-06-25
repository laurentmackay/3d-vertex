import os

import numpy as np
from Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder
from VertexTissue.Stochastic import edge_reaction_selector, reaction_times



from VertexTissue.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.funcs_orig import clinton_timestepper
from VertexTissue.globals import press_alpha, inter_edges_middle, inter_edges_middle_bis, inter_edges_outer, inter_edges_outer_bis, inner_arc, outer_arc, pit_strength, myo_beta, l_apical
from VertexTissue.Dict import closest_dict_value, dict_product, dict_product_nd_shape, last_dict_value
from VertexTissue.Memoization import  function_call_savepath
from VertexTissue.util import arc_to_edges, edge_index, get_myosin_free_cell_edges, inside_arc, find_first
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, edge_crumpler, extension_remodeller, shrink_edges

import VertexTissue.globals as const




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

def invagination_depth2(G):
        basal_offset = G.graph['basal_offset']

        z0 = np.mean([G.node[n]['pos'][-1]  for n in G.neighbors(basal_offset)])

        return np.mean([ G.node[n]['pos'][-1]-z0 for n in belt])

def final_lumen_depth(d):
        return lumen_depth(last_dict_value(d))

def final_depth(d, t_final=None):
        
        if t_final is None:
                return invagination_depth(last_dict_value(d))
        else:
                return  invagination_depth(closest_dict_value(d, t_final))


def final_depth2(d,**kw):
        return invagination_depth2(last_dict_value(d))

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

def final_arc_ratio(d):
        w1=final_width(d)
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
def final_angle(d, **kw):
    # kw['intercalations']=min(kw['intercalations'],6)
    return angle(last_dict_value(d), **kw)  

def final_corner_angle(d, **kw):
    kw['intercalations']=min(kw['intercalations'],6)
    return angle(last_dict_value(d), **kw)  

def run(phi0, remodel=True, press_alpha=const.press_alpha, L0_T1=0.0, verbose=False, belt=True, intercalations=0, outer=False, double=False, viewable=viewable, orig_forces=False):
    
    
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    

    belt = get_outer_belt(G_apical)

    # p10=pit_strength
    # p03=pit_strength
    # m=(p10-p03)/0.7
    # b=p03-0.3*m
    # pit_strength=m*phi0+b

    const.press_alpha=press_alpha
        

    inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

    ec=.2

    k_eff = (phi0-ec)/(1-ec)
    alpha=1
    sigma = (alpha*ec*l_apical*(-1+phi0)+(ec-phi0)*pit_strength*myo_beta)/((-1+ec)*myo_beta)
    # sigma=pit_strength
    t_start = 375




    squeeze = SG.arc_pit_and_intercalation(G, belt, t_1=t_start, inter_edges=inter_edges, t_intercalate=t_start, pit_strength=sigma)

    
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

    clinton_timestep, uncontracted = clinton_timestepper(G, inter_edges)

    
    def label_contracted(i,j):
        inds = edge_index(G, inter_edges)
        uncontracted[ find_first( edge_index(G, (i,j)) == inds )] = False

    # b
    #create integrator
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=blacklist, RK=1,
                                    intercalation_callback=label_contracted,
                                    angle_tol=.01, length_rel_tol=0.05,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None } if viewable else False, minimal=False, **kw)



    integrate(5, 4e4,
              pre_callback=squeeze,
              dt_init=0.5,
              adaptive=True,
              timestep_func=clinton_timestep,
              adaptation_rate=1,
              dt_min=1e-1*k_eff,
              save_rate=100,    
              verbose=verbose,
              save_pattern=pattern,
              resume=True,
              save_on_interrupt=False,
              orig_forces=orig_forces,
              check_forces=True)

intercalations=[0, 4, 6, 8, 12]

intercalations=[0, 4, 6, 8, 12]

def main():
    run(750,  phi0=0.3, cable=True)

phi0s=[1.0,]

L0_T1s=np.linspace(0, l_apical, 10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])
L0_T1s = L0_T1s[L0_T1s<=1.2]
# L0_T1s=np.array([0.0,])
L0_T1s = [L0_T1s[0], (L0_T1s[0]+ L0_T1s[-1])/2, L0_T1s[-1]]
L0_T1s = [ (L0_T1s[-1])/2, L0_T1s[-1], 3*(L0_T1s[-1])/2, 4*(L0_T1s[-1])/2, l_apical]
# L0_T1s = [0.0,]
# L0_T1s=[l_apical, ]

remodel=False

clinton_middle_orig = {'intercalations':intercalations, 'remodel':False, 'L0_T1':l_apical, 'orig_forces':True , 'press_alpha':const.press_alpha*7}

clinton_middle = {'intercalations':intercalations, 'remodel':False, 'L0_T1':l_apical }
clinton_middle_hi_pressure = {'intercalations':intercalations, 'remodel':False, 'L0_T1':l_apical, 'press_alpha':0.046 }
# kws = [*kws_middle, *kws_outer, *kws_double]

if __name__ == '__main__':
    
    def foo(*args):
        pass
#     sweep([1.0], run, kw=clinton_middle_orig, savepath_prefix=base_path, overwrite=True, pre_process=foo)
    run(1.0, L0_T1=l_apical, intercalations=12,   verbose=True, viewable=True, press_alpha=0.046, orig_forces=True)

