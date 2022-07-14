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
    base_path = './data/SAC+127/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC+127/'

from VertexTissue.util import last_item
from VertexTissue.funcs import euclidean_distance


phi0s=[0.3,  .4, .5,  0.6, .7, .8, .9]
colors=['#59e441','#be4ef8','#26b5ec']
if __name__ == '__main__':

    phi_elastic=1

    plt.figure()
    for lvl in (0,1,2):
        c=colors[lvl]
        adv=np.load(f'viscoelastic_peripheral_cable_advantange_127_level_{lvl}.npy')
        adv_no_cable=np.load(f'viscoelastic_peripheral_advantange_127_level_{lvl}.npy')

        
        plt.plot(phi0s, adv_no_cable, label=r'$\alpha_{\rm VE}$ position '+str(lvl+1), linestyle='-', color=c)
        plt.plot(phi0s,adv, label=r'$\alpha_{\rm cable}$', linestyle='--', color=c)

    plt.xlabel('$\phi_0$', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=16)



    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'viscoelastic_peripheral_advantanges_127.pdf',dpi=200)
    plt.show()
    

