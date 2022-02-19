from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import vertex_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1


from VertexTissue.globals import default_ab_linker, default_edge

from VertexTissue.PyQtViz import edge_view

edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()
spoke_attr = default_edge.copy()
def main(dt):

    
        #setup the desired rod attributes
        # spoke_attr['tau'] = tau
        # linker_attr['tau'] = tau
        edge_attr['tau'] = 60
        
        # initialize the tissue
        # G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
        G, G_apical = tissue_3d(  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)

        belt = get_outer_belt(G_apical)

        # edge_view(G, exec=True)


        #initialize some things for the callback
        intercalation = T1.simple_T1(G)
        invagination = SG.arcs_and_pit(G,belt)
        


        #create integrator
        integrate = vertex_integrator(G, G_apical, pre_callback=invagination, player=False, viewer=True, maxwell=True, minimal=False)
        #integrates
        # try:
        integrate(dt, 500, 
                dt_init = 1e-3,
                adaptive=True,
                dt_min=1e-2,
                save_rate=dt,
                length_prec=0.01,
                save_pattern=None)
        # except:
        #     print(f'failed to integrate tau={tau}')
        #     pass


if __name__ == '__main__':
    main(1)

    print('done')
