from asyncio.subprocess import Process
from distutils import extension
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1


from VertexTissue.globals import default_ab_linker, default_edge, l_apical

from VertexTissue.PyQtViz import edge_view

edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()
spoke_attr = default_edge.copy()
visco=False
if visco:
        edge_attr={'l_rest':l_apical,'l_rest_0':l_apical,'tension':0.0,'tau':360,'myosin':0.0}
        spoke_attr = edge_attr.copy()

def main(dt):

    
        #setup the desired rod attributes
        # spoke_attr['tau'] = 60
        # linker_attr['tau'] = tau
        edge_attr['tau'] = 60
        
        # initialize the tissue
        # G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
        G, G_apical = tissue_3d(  basal=False, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)

        belt = get_outer_belt(G_apical)

        # edge_view(G, exec=True)


        #initialize some things for the callback
        basal=False
        invagination = SG.arcs_with_intercalation(G,belt, basal=basal)
        extension = SG.convergent_extension_test(G)
        # invagination = SG.just_intercalation(G,belt)


        #create integrator
        integrate = monolayer_integrator(G, G_apical, pre_callback=extension, player=False, viewer={'draw_axes':False,  'attr':'myosin'}, maxwell=visco, minimal=False, blacklist=False, ndim=2, SLS=visco, tension_remodel=visco)
        #integrates
        # try:
        integrate(dt, 45000, 
                dt_init = 1e-3,
                adaptive=True,
                dt_min=1e-3,
                save_rate=1,
                length_prec=0.01,
                save_pattern='./data/viscoelastic/intercalation_compress_*.pickle')
        # except:
        #     print(f'failed to integrate tau={tau}')
        #     pass


if __name__ == '__main__':
    main(10)

    print('done')
