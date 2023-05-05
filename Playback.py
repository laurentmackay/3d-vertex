import os
import pickle
import numpy as np

from VertexTissue.Player import pickle_player
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.util import  finder, shortest_edge_network_and_time
from VertexTissue.Dict import last_dict_value

def isfinite(x):
    return np.isfinite(x)*1.0


def shortest_edge_network_annotated(d):

    G,_= shortest_edge_network_and_time(d)
    # G=last_dict_value(d)
    centers = G.graph['centers']
    edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]
    # edges=G.edges
    for e in G.edges:
        a,b=e
        if e in edges:
            G[a][b]['len'] = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/3.4
        else:
            G[a][b]['len'] = np.nan
    return G

def final_network_annotated(d):

    # G,_= shortest_edge_network_and_time(d)
    G=last_dict_value(d)
    centers = G.graph['centers']
    edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]
    edges=G.edges
    for e in G.edges:
        a,b=e
        if e in edges:
            G[a][b]['len'] = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/3.4
        else:
            G[a][b]['len'] = np.nan
    return G

def network_strain(G):

    # G,_= shortest_edge_network_and_time(d)
    # G=last_dict_value(d)
    # centers = G.graph['centers']
    # edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]
    # edges=G.edges
    for e in G.edges:
        a,b=e
       
        G[a][b]['strain'] = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/G[a][b]['l_rest']-1

    return G

from VertexTissue.globals import pit_centers

def delta_network_strain(G0):

    G0=network_strain(G0)
    n0=len(G0.nodes)
    def inner(G):
        arc_strain=[]
        for e in G.edges:
            a,b=e
            strain = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/G[a][b]['l_rest']-1
            G[a][b]['strain'] = strain
            a0 = a % n0
            b0 = b % n0
            delta = strain - G0[a0][b0]['strain']
            G[a][b]['delta_strain'] = delta
            if G[a][b]['myosin'] > 0:
                
                G[a][b]['delta_strain_myo'] = delta
        #         if (not np.any([ n in pit_centers for  n in G.neighbors(a)])) and  (not np.any([ n in pit_centers for  n in G.neighbors(b)])):
        #             arc_strain.append(delta)

        # print(np.mean(arc_strain))
        return G

    return inner
# 'data/elastic/Task3_T1_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
forces=np.linspace(0,600,80)
# pickle_player(path='./data/SAC_127/', pattern=f'triple_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)

# pickle_player(path='./data/Step1/run', pattern=f'L0_T1=3.4_0.8.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None)

# pickle_player(path='./data/Step1/run', pattern=f'outer_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None, pre_process=last_dict_value)

# pickle_player(path='./data/Step2_bis/run', pattern=f'*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None, pre_process=final_network_annotated)


# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_intercalations=10_outer_0.4.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)


# pickle_player(path='./data/energy_testing/run', pattern=f'visco_level=1_continuous_pressure_intercalations=8_t_final=800_pit_strength=300_basal_press_alpha=0.046_1000.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

# pickle_player(path='./data/Step5/run', pattern=f'no_remodel_L0_T1=3.4_pit_strength=157.89473684210526_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)


# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec=0.0_no_pit_T1s_SLS_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)


pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec=0.0_no_pit_T1s_SLS_0.2.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

import networkx as nx
def splice_save_dicts(d1, d2, join=nx.disjoint_union, join_kws={}):
    d={}
    t1=np.fromiter(d1.keys(),dtype=float)
    t2=np.fromiter(d2.keys(),dtype=float)

    t2_inds = finder(t1,t2)

    for a,b in t2_inds:
        a=int(a)
        b=int(b)
        t_avg = (t1[a]+t2[b])/2.0
        d[t_avg] = join(d1[t1[a]], d2[t2[b]], **join_kws)
    
    return d

def join_graphs(G1, G2, displacement=None):
    centers_1, centers_2 = G1.graph['centers'], G2.graph['centers']
    if displacement is not None:
        for n in G2.nodes:
            G2.node[n]['pos']+=displacement
    circum_sorted_1, circum_sorted_2 = G1.graph['circum_sorted'], G2.graph['circum_sorted']

    G = nx.disjoint_union(G1, G2)
    offset = len(G1.nodes)-1
    G.graph['centers'] = np.concatenate((centers_1, centers_2 + offset))
    G.graph['circum_sorted'] = [*circum_sorted_1, *[s+offset+1 for s in circum_sorted_2] ]
    return G


def belt_z(G, belt):

        return np.mean([ G.node[n]['pos'][-1] for n in belt])

def load_pickle(path=os.getcwd(), file=''):
    with open(os.path.join(path,file), 'rb') as input:
        return pickle.load(input)
ec=0.09444444444444444

d0 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_1.0.pickle')
d1 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle')
d2 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle')
d3 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle')


G, G_apical = tissue_3d( hex=7,  basal=True)


belt = get_outer_belt(G_apical)

z1=belt_z(last_dict_value(d1), belt)
z2=belt_z(last_dict_value(d2), belt)
z3=belt_z(last_dict_value(d3), belt)
z0=belt_z(last_dict_value(d0), belt)

d = splice_save_dicts(d1,d2, join=join_graphs, join_kws={'displacement': np.array([65.0,0.0,-z2+z1])})
d = splice_save_dicts(d,d3, join=join_graphs, join_kws={'displacement':np.array([123.0,0.0,-z3+z1])})
# d = splice_save_dicts(d,d0, join=join_graphs, join_kws={'displacement':np.array([-40.0,0.0,-z0+z1])})

pickle_player(save_dict=d, attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, pre_process=delta_network_strain(last_dict_value(d0)))

exit()

pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec=0.09444444444444444_extend_no_T1_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, pre_process=network_strain)
# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_intercalations=16_0.5.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

pickle_player(path='./data/Step4/run', pattern=f'no_remodel_L0_T1=3.4_intercalations=8_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, )

# pickle_player(path='./data/Step2_stochastic_no_arches/run', pattern=f'L0_T1=3.4_intercalations=24_outer_double_ndim=2_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

# pickle_player(path='./data/Step2_bis/run', pattern=f'intercalations=8_outer_0.6.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

# pickle_player(path='./data/SAC_pit/', pattern=f'peripheral_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)
# pickle_player(path='./data/SAC_127/', pattern=f'peripheral_visco_phi0=0.6_1500.0.pickle', attr='myosin', speedup=5, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, start_time=0.0)


# pickle_player(path='./data/SAC_127/', pattern=f'peripheral_outer_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, start_time=None, nodeLabels=None)

# pickle_player(path='./data/SAC_127/', pattern=f'visco_phi0=0.5_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, nodeLabels=None)


# pickle_player(path='./data/mini+127/', pattern=f'visco_phi0=*_435.8974358974359.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=last_item, check_timestamp=False, start_time=0.3)

