import os

import numpy as np



from VertexTissue.Sweep import sweep



import VertexTissue.SG as SG


from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.globals import belt_strength, outer_arc, inner_arc, pit_strength
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.Dict import dict_product, first_dict_value, last_dict_value
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.visco_funcs import crumple, extension_remodeller, shrink_edges






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


middle_edge = [234, 233]
outer_edge = [364, 363]
# def buckle_angle(G, e=[234, 233]):
#         a=e[1]-1
#         nhbrs = np.sort(list(G.neighbors(a)))
#         nhbrs = np.array([n for n in nhbrs if n not in e])
#         lr = nhbrs[[np.sum([nn in nhbrs for  nn in G.neighbors(n)])==1 for n in nhbrs]]

#         a=G.node[a]['pos']
#         b=G.node[e[0]]['pos']
#         c=G.node[e[1]]['pos']
#         d=G.node[lr[0]]['pos']
#         e=G.node[lr[1]]['pos']

#         ab=unit_vector(a,b)
#         bc=unit_vector(b,c)
#         bd=unit_vector(b,d)
#         be=unit_vector(b,e)
#         normal = (np.cross(ab,bd)+np.cross(be,ab))/2

#         bc = bc-np.dot(bc, normal)*normal
#         bc /= np.linalg.norm(bc)

#         return np.arccos(np.dot(ab,bc))

def buckle_angle_finder(G, edge=None, basal=False):
    centers=G.graph['centers']

    if basal:
        centers = centers + G.graph['basal_offset']
        
    centers=set(centers)
    # node_dists = [euclidean_distance(G.node[n]['pos'], G.node[0]['pos']) for n in edge]

    n0 = set(G.neighbors(edge[0]))
    n1 = set(G.neighbors(edge[1]))
    center_nhbrs = list(set.union(set.difference(set.intersection(n1, centers),set.intersection(n0, centers) ), set.difference(set.intersection(n0, centers),set.intersection(n1, centers) )))
    
    # center_dists = [euclidean_distance(G.node[c]['pos'], G.node[e0]['pos']) for c in center_nhbrs]
    c0 = center_nhbrs[0]
    c1 = center_nhbrs[1]

    e0=[e for e in edge if e in G.neighbors(c0)][0]
    e1=[e for e in edge if e in G.neighbors(c1)][0]
    # nhbrs = np.sort(list(G.neighbors(c1)))
    # nhbrs = np.array([n for n in nhbrs if n not in edge])
    # lr = nhbrs[[np.sum([nn in nhbrs for  nn in G.neighbors(n)])==1 for n in nhbrs]]


    def inner(G):

        a=G.node[c1]['pos']
        b=G.node[e1]['pos']
        c=G.node[e0]['pos']
        d=G.node[c0]['pos']
        # d=G.node[lr[0]]['pos']
        # e=G.node[lr[1]]['pos']

        ab=unit_vector(a,b)
        bc=unit_vector(b,c)
        cd=unit_vector(c,d)
        straight = (ab+cd)/np.linalg.norm(ab+cd)
        return np.arccos(np.dot(bc, straight))
        # bd=unit_vector(b,d)
        # be=unit_vector(b,e)
        # normal = (np.cross(ab,bd)+np.cross(be,ab))/2

        # bc = bc-np.dot(bc, normal)*normal
        # bc /= np.linalg.norm(bc)

        # return np.arccos(np.dot(ab,bc))
        
    return inner

def max_buckle_angle(d, edge = None):
    max_angle=0.0
    G=first_dict_value(d)
    buckle_angle = buckle_angle_finder(G, edge=edge)
    for G in d.values():
        max_angle=max(buckle_angle(G),max_angle)
    
    return max_angle

def final_buckle_angle(d, edge = None):
    G=last_dict_value(d)
    buckle_angle = buckle_angle_finder(G, edge=edge)
    max_angle=buckle_angle(G)
    
    return max_angle

def run(phi0, remodel=True, cable=True, L0_T1=0.0, inner_only=False, pit_before=False, pit_after=False, verbose=False, belt=False, outer=False):
    
    if not outer:
        inter_edges = [middle_edge, ]
    else:
        inter_edges = [outer_edge, ]
    
    
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')
    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    if belt:
        belt = get_outer_belt(G_apical)
    else:
        belt = []


    if inner_only:
        arcs=(inner_arc,)
    else:
        arcs=(outer_arc, inner_arc)

    t_start = 375.0 if pit_before else 0.0
    squeeze0 = SG.arc_pit_and_intercalation(G, belt, arcs=arcs,
                                         arc_strength=belt_strength if cable else 0.0, belt_strength=belt_strength if len(belt)>0 else 0, t_1=t_start,
                                         inter_edges=inter_edges, t_intercalate=t_start,  intercalation_strength=750,
                                          pit_strength=pit_strength if pit_before else 0)

    squeeze1 = SG.arc_pit_and_intercalation(G, belt, arcs=arcs,
                                         arc_strength=belt_strength if cable else 0.0, belt_strength=belt_strength if len(belt)>0 else 0, t_1=0,
                                          inter_edges=inter_edges, t_intercalate=0,  intercalation_strength=0,
                                          pit_strength=pit_strength if pit_after else 0)
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
    
    shrink = shrink_edges(G, L0=L0_T1)



    def shrink_edges_and_terminate(node, neighbour):
        terminate()
        shrink(node, neighbour)
    
    #create integrator
    integrate = monolayer_integrator(G, G_apical,
                                    blacklist=True, RK=1,
                                    intercalation_callback=shrink_edges_and_terminate,
                                    termination_callback=wait_for_intercalation,
                                    angle_tol=.05, length_rel_tol=0.1,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)



    integrate(5, 3000,
              pre_callback=squeeze0,
              dt_init=1e-3,
              adaptive=True,
              dt_min=1e-6,
              save_rate=50,
              verbose=verbose,
              save_pattern=None)

    buckle_angle    = buckle_angle_finder(G, edge=inter_edges[0])

    def is_buckled(t):
        

        if t>5000:
            angle = (180/np.pi)*buckle_angle(G)
            return angle>45
        else:
            return False

    integrate(5, 15000,
              dt_init=1e-3,
              adaptive=True,
              dt_min=1e-1,
              termination_callback=is_buckled, pre_callback=squeeze1,
              save_rate=50,
              verbose=verbose,
              save_pattern=pattern)

def main():
    run(750,  phi0=0.3, cable=True)

phi0s=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

L0_T1s=np.linspace(0,3.4,10)
L0_T1s = np.unique([*np.linspace(0,L0_T1s[2],6), *L0_T1s])

if __name__ == '__main__':
    # main()
    
    # kws = dict_product({'cable': [True,False],
    #                     'L0_T1': L0_T1s,
    #                     'pit_before': [False, True],
    #                     'pit_after': [False,True],
    #                     'belt': [True, False]})

    kws = dict_product({'L0_T1': L0_T1s,  'outer':False, 'remodel':True, 'cable': True, 'belt': True, 'inner_only': False})


    sweep(phi0s, run, kw=kws, savepath_prefix=base_path, overwrite=False)
    # run(1.0, L0_T1=0, outer=True)

