import numpy as np

from VertexTissue.Player import pickle_player
from VertexTissue.funcs import euclidean_distance
from VertexTissue.util import imin, last_item, shortest_edge_network_and_time

def isfinite(x):
    return np.isfinite(x)*1.0


def shortest_edge_network_annotated(d):

    G,_= shortest_edge_network_and_time(d)
    centers = G.graph['centers']
    edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]

    for e in G.edges:
        a,b=e
        if e in edges:
            G[a][b]['len'] = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/3.4
        else:
            G[a][b]['len'] = np.nan
    return G
# 'data/elastic/Task3_T1_{fmag}_{tau}_eta_{const.eta}_dt_{dt}_*.pickle'
forces=np.linspace(0,600,80)
pickle_player(path='./data/SAC_127/', pattern=f'triple_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)
# pickle_player(path='./data/SAC_pit/', pattern=f'peripheral_visco_phi0=0.3_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False,  nodeLabels=None)
# pickle_player(path='./data/SAC_127/', pattern=f'peripheral_visco_phi0=0.6_1500.0.pickle', attr='myosin', speedup=5, cell_edges_only=True, apical_only=True, check_timestamp=False, nodeLabels=None, start_time=0.0)


# pickle_player(path='./data/SAC_127/', pattern=f'triple_elastic_edges_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, start_time=None, nodeLabels=None)

# pickle_player(path='./data/SAC_127/', pattern=f'visco_phi0=0.5_*.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=shortest_edge_network_annotated, check_timestamp=False, nodeLabels=None)


# pickle_player(path='./data/mini+127/', pattern=f'visco_phi0=*_435.8974358974359.pickle', attr='myosin', speedup=0, cell_edges_only=True, apical_only=True, pre_process=last_item, check_timestamp=False, start_time=0.3)

