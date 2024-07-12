import os
import pickle
import numpy as np

from VertexTissue.Player import pickle_player
from VertexTissue.Tissue import get_outer_belt, tissue_3d
from VertexTissue.util import  finder, shortest_edge_network_and_time

from ResearchTools.Dict import last_dict_value
from ResearchTools.Geometry import euclidean_distance

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

def belt_z(G, belt):
        return np.mean([ G.node[n]['pos'][-1] for n in belt])


ec=0.09444444444444444   
# d0 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_1.0.pickle')
# d1 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle')
# d2 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle')
# d3 = load_pickle(path='./data/Step2_bis/run', file=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle')

# G0=last_dict_value(d0)

# delta_strain = delta_network_strain(G0)
insta_save = True

G, G_apical = tissue_3d( hex=7,  basal=True)
belt = get_outer_belt(G_apical)

# z1=belt_z(last_dict_value(d1), belt)
# z2=belt_z(last_dict_value(d2), belt)
# z3=belt_z(last_dict_value(d3), belt)
# z0=belt_z(last_dict_value(d0), belt)

style={'attr':'myosin','check_timestamp':False, 'nodeLabels':None, 'vmin':-0.55, 'vmax':0.55, 
       'start_time':0, 'insta_save':insta_save}

pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle',
              distance=150, elevation=4, azimuth=0,
              center=(0,0,-8.7), render_dimensions=(1350,1440), scale_factor=5,
              filename='contraction_apical_strain', apical_only=True, **style)

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle',
#               distance=150, elevation=4, azimuth=0, 
#               center=(0, 0, -7.2),  render_dimensions=(1350,1440), scale_factor=5,
#               filename='extension_apical_strain', apical_only=True, **style)

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle',
#               distance=150, elevation=3, azimuth=0, 
#               center=(0,0,-6.2), render_dimensions=(1350,1440), scale_factor=5,
#               filename='symmetric_apical_strain', apical_only=True, **style)


# style['edgewidth']=80

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle',
#               distance=28, elevation=90, azimuth=0,
#               center=(0,0,-8.7), render_dimensions=(1080,720), scale_factor=3,
#               filename='contraction_pit_strain', apical_only=True, **style)

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle',
#               distance=20, elevation=90, azimuth=0, 
#               center=(0, 0, -7.2), render_dimensions=(1080,720), scale_factor=3,
#               filename='extension_pit_strain', apical_only=True, **style)

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle',
#               distance=9.4, elevation=90, azimuth=0, 
#               center=(0,0,-6.2), render_dimensions=(1080,720), scale_factor=3,
#               filename='symmetric_pit_strain', apical_only=True, **style)

style['edgewidth']=80

# style['vmin']=-0.45
# style['vmax']=0.45
# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_no_T1_0.3.pickle',
#               distance=288, elevation=5, azimuth=0,
#               center=(0,0,-16.6),  render_dimensions=(1350,1080), scale_factor=5,
#               filename='contraction_basal_strain', basal=True, apical=False, **style)


# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_contract_no_T1_0.3.pickle',
#               distance=288, elevation=5, azimuth=0,
#               center=(0,0,-16.6),  render_dimensions=(1350,1080), scale_factor=5,
#               filename='extension_basal_strain', basal=True, apical=False, **style)

# pickle_player(path='./data/Step2_bis/run', pattern=f'no_remodel_L0_T1=3.4_no_scale_pit_ec={ec}_extend_no_T1_0.3.pickle',
#               distance=288, elevation=6, azimuth=0, 
#               center=(0,0,-12.6),  render_dimensions=(1350,1080), scale_factor=5,
#               filename='symmetric_basal_strain', basal=True, apical=False, **style)

