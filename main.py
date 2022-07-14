from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1


from VertexTissue.globals import default_ab_linker, default_edge, l_apical, basal_offset, inter_edges, arc1, arc2, arc3, arc4, arc5, arc6

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
        # edge_attr['tau'] = 60
        
        # initialize the tissue
        # G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
        G, G_apical = tissue_3d(  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)

        belt = get_outer_belt(G_apical)
        
        # edge_view(G, exec=True)
        def shrink_edges(node,neighbour,basal=True):
                G[node][neighbour]['l_rest']=0
                if basal:
                        G[node+basal_offset][neighbour+basal_offset]['l_rest']=0

        ec=0.3
        def extension_remodelling(ell,L):
                eps=0
                if L>0:
                        eps=(ell-L)/L
                
                if (L<=0 and ell>0) or eps>ec:
                        return (ell-L)-ec*max(L,0)
                else:
                        return 0

        #initialize some things for the callback
        basal=False
        # invagination = SG.arcs_with_intercalation(G,belt, basal=basal)
        invagination = SG.arc_pit_and_intercalation(G, belt, basal_intercalation=basal, arcs=(arc1, arc2, arc3, arc4, arc5, arc6),  intercalation_strength=0, pit_strength=0.001, belt_strength=0, arc_strength=0)
        # invagination = SG.just_intercalation(G,belt)


        #create integrator
        integrate = monolayer_integrator(G, G_apical, pre_callback=invagination, intercalation_callback=shrink_edges, player=False, viewer={'draw_axes':False, 'nodeLabels':None}, maxwell=visco, minimal=False, blacklist=True, SLS=visco, tension_remodel=visco, maxwell_nonlin=extension_remodelling)
        #integrates
        # try:
        prefix = 'visco' if visco else 'elastic'
        integrate(dt, 10000, 
                dt_init = 1e-3,
                adaptive=True,
                dt_min=1e-1,
                save_rate=1,
                length_prec=0.01,
                save_pattern=f'./data/{prefix}/S{len(inter_edges)}_zero_rest_outer_*.pickle')
        # except:
        #     print(f'failed to integrate tau={tau}')
        #     pass


if __name__ == '__main__':
    main(10)

    print('done')
