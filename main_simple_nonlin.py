from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1


from VertexTissue.globals import default_ab_linker, default_edge, l_apical, basal_offset, inter_edges
import VertexTissue.globals as const
from VertexTissue.PyQtViz import edge_view

edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()
spoke_attr = default_edge.copy()

visco=True
if visco:
        edge_attr={'l_rest':l_apical,'l_rest_0':l_apical,'tension':0.0,'tau':60,'myosin':0.0}
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

        def soften_shoulders(node, neighbour, cents,basal=True):
                # G[node][neighbour]['l_rest']=0
                node_links = (set(list(G.neighbors(cents[1]))) & set(list(G.neighbors(node))) ) | set([cents[1]]) 
                nhbr_links = (set(list(G.neighbors(cents[0]))) & set(list(G.neighbors(node))) ) | set([cents[0]]) 

                for l in node_links:
                        const.softened.append((node, l))
                        const.softened.append((l, node))

                for l in nhbr_links:
                        const.softened.append(( neighbour, l))
                        const.softened.append((l,  neighbour))
                # print(node)
                # print(neighbour)
                # print(G.neighbors(cents[0]))



                # if basal:
                #         G[node+basal_offset][neighbour+basal_offset]['l_rest']=0


        phi0=.6

        # const.pit_strength=400
        # L0=l_apical

        def l_rest(ell,e,ec=.2):

                L0=G[e[0]][e[1]]['l_rest']

                if ell<(1-ec)*L0:
                      return (1-phi0)*ell/(1-ec)+ phi0*L0
                else:
                        return L0


        def l_rest_shoulder(ell,e,ec=.2):

                L0=G[e[0]][e[1]]['l_rest']

                if ell<(1-ec)*L0 and e in const.softened:
                        print('we got a soft boi over here')
                        return (1-phi0)*ell/(1-ec)+ phi0*L0
                else:
                        return L0

        
        def extension_remodelling(ell,L, ec=.3):
                eps=0
                if L>0:
                        eps=(ell-L)/L
                
                if (L<=0 and ell>0) or (eps>ec and L<3.4):
                        val=(ell-L)-ec*max(L,0)
                        print(f'extending an edge with rest length {L} at a rate {val}')
                        return val
                else:
                        return 0

        #initialize some things for the callback
        basal=False
        # invagination = SG.arcs_with_intercalation(G,belt, basal=basal)
        invagination = SG.arc_pit_and_intercalation(G, belt, basal_intercalation=basal, intercalation_strength=1500 if visco else 1000)
        # invagination = SG.just_intercalation(G,belt)


        #create integrator
        integrate = monolayer_integrator(G, G_apical,
                                        pre_callback=invagination,
                                        # intercalation_callback=soften_shoulders,
                                        player=False,
                                        viewer={'draw_axes':False, 'nodeLabels':None},
                                        minimal=False,
                                        blacklist=True,
                                        maxwell=False,
                                        # maxwell_nonlin=extension_remodelling, maxwell=True,
                                        rest_length_func=l_rest)
        # )
        #integrates
        # try:
        prefix = 'visco' if visco else 'elastic'
        integrate(dt, 10000, 
                dt_init = 1e-3,
                adaptive=True,
                dt_min=1e-1,
                save_rate=1,
                length_prec=0.01,
                save_pattern=f'./data/dicts/S5_nonlinear_spring.pickle')
        # except:
        #     print(f'failed to integrate tau={tau}')
        #     pass


if __name__ == '__main__':
    main(10)

    print('done')
