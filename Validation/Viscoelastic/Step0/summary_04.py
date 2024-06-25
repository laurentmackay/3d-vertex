import os

from matplotlib import colors
print('this is some basic output')

from doctest import FAIL_FAST
from itertools import product
from pathos.pools import ProcessPool as Pool
import numpy as np


from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, T1_shell, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc
import VertexTissue.globals as const

from VertexTissue.util import first

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/SAC+_127/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC+_127/'

from VertexTissue.Dict import last_dict_value
from VertexTissue.Geometry import euclidean_distance


phi0s=[0.3,  .4, .5,  0.6, .7, .8, .9]
colors=['#59e441','#be4ef8','#26b5ec','#e09400','#fb8def']

if __name__ == '__main__':

    phi_elastic=1

    plt.figure()
    for lvl in (0,1,2,3):
        c=colors[lvl]
        adv=np.load(f'viscoelastic_cable_advantange_triple_127_level_{lvl}.npy')
        adv_no_cable=np.load(f'viscoelastic_advantange_triple_127_level_{lvl}.npy')

        
        plt.plot([ *phi0s, 1.0], [*(adv_no_cable), 0.0] , label=r'position '+str(lvl+1), linestyle='-', color=c)
        plt.plot([ *phi0s, 1.0],[*(adv), adv[-1]],  linestyle='--', color=c)
        # plt.plot(phi0s,adv+adv_no_cable,  linestyle='-.', color=c)
    # plt.plot([ *phi0s, 1.0], [*(1-np.array(phi0s)), 0.0] , label=r'best case scenario', linestyle='--', color='k')

    plt.xlabel('$\delta$', fontsize=16)
    plt.ylabel(r'reduction in contraction force (non-dim)', fontsize=14)
    plt.ylim(0,1.05)


    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'viscoelastic_triple_contraction_forces_127.png',dpi=200)
    plt.show()
    

