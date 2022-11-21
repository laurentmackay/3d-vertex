import numpy as np

from VertexTissue.Player import pickle_player
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.util import  shortest_edge_network_and_time
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
# 'data/elastic/Task3_T1_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
forces=np.linspace(0,600,80)
# pickle_player(path='./data/SAC_127/', pattern=f'triple_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)

# pickle_player(path='./data/Step1/run', pattern=f'L0_T1=3.4_0.8.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None)

# pickle_player(path='./data/Step1/run', pattern=f'outer_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None, pre_process=last_dict_value)

# pickle_player(path='./data/Step2_bis/run', pattern=f'*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False,  nodeLabels=None, pre_process=final_network_annotated)

pickle_player(path='./data/Step2_stochastic_no_arches/run', pattern=f'L0_T1=3.4_intercalations=24_outer_double_ndim=2_1.0.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

# pickle_player(path='./data/Step2_bis/run', pattern=f'intercalations=8_outer_0.6.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None)

# pickle_player(path='./data/SAC_pit/', pattern=f'peripheral_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)
# pickle_player(path='./data/SAC_127/', pattern=f'peripheral_visco_phi0=0.6_1500.0.pickle', attr='myosin', speedup=5, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, start_time=0.0)


# pickle_player(path='./data/SAC_127/', pattern=f'peripheral_outer_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, start_time=None, nodeLabels=None)

# pickle_player(path='./data/SAC_127/', pattern=f'visco_phi0=0.5_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, nodeLabels=None)


# pickle_player(path='./data/mini+127/', pattern=f'visco_phi0=*_435.8974358974359.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=last_item, check_timestamp=False, start_time=0.3)

