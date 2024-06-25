from asyncio.subprocess import Process
from operator import truediv
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1


from VertexTissue.globals import default_ab_linker, default_edge

from VertexTissue.PyQtViz import edge_view


dt=0.1
taus = np.logspace(6,1,5)
def run(tau):

    edge_attr = default_edge.copy()
    linker_attr = default_ab_linker.copy()
    spoke_attr = default_edge.copy()

    #setup the desired rod attributes
    # spoke_attr['tau'] = tau
    # linker_attr['tau'] = tau
    edge_attr['tau'] = tau
    
    # initialize the tissue
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
    # G, G_apical = tissue_3d(  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)

    belt = get_outer_belt(G_apical)

    # edge_view(G, exec=True)


    #initialize some things for the callback
    intercalation = T1.simple_T1(G)
    # invagination = SG.arcs_and_pit(G,belt)
    


    #create integrator
    integrate = monolayer_integrator(G, G_apical, pre_callback=intercalation, player=False, viewer=False, maxwell=truediv, minimal=False)
    #integrates
    # try:
    integrate(dt, 1000, 
            dt_init = 1e-3,
            adaptive=False,
            dt_min=0,
            save_rate=1,
            length_prec=0.01,
            save_pattern='./data/T1/viscoelastic_edges_{tau}_*.pickle')
    # except:
    #     print(f'failed to integrate tau={tau}')
    #     pass


if __name__ == '__main__':
    
    Pool(nodes=6).map( run, taus)

    print('done')
