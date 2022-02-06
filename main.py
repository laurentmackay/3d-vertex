from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import vertex_integrator
from VertexTissue.Tissue import tissue_3d, get_outer_belt
import VertexTissue.SG as SG

from VertexTissue.globals import default_ab_linker, default_edge
edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()
spoke_attr = default_edge.copy()
def main(tau):

    
        #setup the desired rod attributes
        # spoke_attr['tau'] = tau
        # linker_attr['tau'] = tau
        edge_attr['tau'] = tau
        
        # initialize the tissue
        G, G_apical = tissue_3d(cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
        belt = get_outer_belt(G_apical)

        #initialize some things for the callback
        invagination = SG.arcs_and_pit(G, belt)
        # invagination = SG.arcs_with_intercalation(G, belt)
        # viewer = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)
        t_last = 0 
        t_plot = 1



        #create integrator
        integrate = vertex_integrator(G, G_apical, pre_callback=invagination, player=False, viewer=True, maxwell=True)
        #integrate
        # try:
        integrate(10.0, 2000, dt_init = 0.001, adaptive=True, save_rate=1, strain_thresh=.01, length_prec=0.01, save_pattern=f'data/testing/adaptive_tau_{tau}_*.pickle')
        # except:
        #     print(f'failed to integrate tau={tau}')
        #     pass


if __name__ == '__main__':
    main(60)

    print('done')
