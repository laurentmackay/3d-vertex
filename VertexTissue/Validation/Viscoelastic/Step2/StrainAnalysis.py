import os
import pickle
import numpy as np

from VertexTissue.Player import pickle_player
from VertexTissue.Geometry import euclidean_distance
from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.util import  finder, shortest_edge_network_and_time
from VertexTissue.Dict import last_dict_value

def load_pickle(path=os.getcwd(), file=''):
    with open(os.path.join(path,file), 'rb') as input:
        return pickle.load(input)
    




def delta_network_strain(G0):

    def network_strain(G):
        for e in G.edges:
            a,b=e
            G[a][b]['strain'] = euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos'])/G[a][b]['l_rest']-1

        return G


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

ec=0.09444444444444444   
d0 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_1.0.pickle')
G0=last_dict_value(d0)

delta_strain = delta_network_strain(G0)


# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle',
#               attr='delta_strain', speedup=0, cell_edges_only=True, apical_only=True, 
#               check_timestamp=False, nodeLabels=None, pre_process=delta_strain,
#               distance=58, elevation=7, azimuth=0,
#               center=(0,0,-2.7),
#               filename='contraction_apical_strain' )

pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle',
              attr='delta_strain', speedup=0, cell_edges_only=True, apical_only=True, 
              check_timestamp=False, nodeLabels=None, pre_process=delta_strain,
              distance=46, elevation=8, azimuth=0, 
              center=(0,0,-5.5),
              filename='extension_apical_strain')

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle',
#               attr='delta_strain', speedup=0, cell_edges_only=True, apical_only=True, 
#               check_timestamp=False, nodeLabels=None, pre_process=delta_strain,
#               distance=46, elevation=8, azimuth=0, 
#               center=(0,0,-5),
#               filename='symmetric_apical_strain')