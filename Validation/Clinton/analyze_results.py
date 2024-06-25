import numpy as np

from VertexTissue.Dict import last_dict_value
from VertexTissue.util import collate_pickles
from VertexTissue.Tissue import get_outer_belt, tissue_3d

import matplotlib.pyplot as plt

d_4  = collate_pickles(path='./data/clinton_og/', pattern='inter_4_t_*.pickle', step=50)
d_6  = collate_pickles(path='./data/clinton_og/', pattern='inter_6_t_*.pickle', step=50)
d_8  = collate_pickles(path='./data/clinton_og/', pattern='inter_8_t_*.pickle', step=50)
d_12 = collate_pickles(path='./data/clinton_og/', pattern='inter_12_t_*.pickle', step=50)

_, G_apical = tissue_3d( hex=7,  basal=True)
belt = get_outer_belt(G_apical)

def invagination_depth(G):
        basal_offset = G.graph['basal_offset']

        z0 = np.mean([G.node[n]['pos'][-1]  for n in G.neighbors(basal_offset)])

        return np.mean([ G.node[n+basal_offset]['pos'][-1]-z0 for n in belt])

def final_depth(d):
        return invagination_depth(last_dict_value(d))


inters=[4, 6, 8, 12]
depths = [final_depth(d) for d in [d_4, d_6, d_8, d_12]]

print(depths)

plt.plot(inters, depths)
plt.show()