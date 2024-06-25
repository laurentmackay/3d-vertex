from itertools import product
import sys


import numpy as np
from pathos.pools import ProcessPool as Pool

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import tissue_3d, get_outer_belt
import VertexTissue.SG as SG

from VertexTissue.globals import default_ab_linker, default_edge

dev_null = open('/dev/null', 'w')
original_stdout = sys.stdout
sys.stdout = dev_null

def main(args):

    tau, type = args

    edge_attr = default_edge.copy()
    linker_attr = default_ab_linker.copy()
    spoke_attr = default_edge.copy()
        
    #setup the desired rod attributes
    if type == 0:
        edge_attr['tau'] = tau
        save_pattern=f'data/testing/viscoelastic_edges_only_{tau}_*.pickle'
    elif type == 1:
        edge_attr['tau'] = tau
        spoke_attr['tau'] = tau
        save_pattern=f'data/testing/viscoelastic_edges_and_spokes_{tau}_*.pickle'
    elif type == 2:
        edge_attr['tau'] = tau
        spoke_attr['tau'] = tau
        linker_attr['tau'] = tau
        save_pattern=f'data/testing/viscoelastic_full_{tau}_*.pickle'


    
    print(f'Starting: tau={tau:.2e} type={type}', file=original_stdout)

    

    #initialize the tissue
    G, G_apical = tissue_3d(cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
    belt = get_outer_belt(G_apical)

    #create the callback
    invagination = SG.arcs_and_pit(G, belt)


    #create integrator
    integrate = monolayer_integrator(G, G_apical, pre_callback=invagination, player=False, maxwell=True)

    #integrate
    try:
        integrate(0.5, 2000, save_rate=1, save_pattern=save_pattern)
        print(f'Done: tau={tau:.2e} type={type}', file=original_stdout)
    except Exception as e:

        print(f'###############\n###############\n failed to integrate tau={tau} \n {e} \n###############\n###############', file=original_stdout)

        pass
    


if __name__ == '__main__':

    pool = Pool(nodes=4)
    # taus =  (6000, 60000, 600000)
    taus =( *6*np.logspace(3, 9, num=7, dtype=int), )
    types = (0,1,2)
    conds =  list(product(taus, types))
    pool.map( main, conds )

    print('All Done!')
