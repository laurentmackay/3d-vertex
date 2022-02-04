from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
from VertexTissue.vertex_3d import vertex_integrator
from VertexTissue.Tissue import tissue_3d, get_outer_belt
import VertexTissue.SG as SG

from VertexTissue.globals import default_ab_linker, default_edge
edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()

def main(tau):

    
        # initialize the tissue

        linker_attr['tau'] = tau
        edge_attr['tau'] = tau

        G, G_apical = tissue_3d(cell_edge_attr=edge_attr, linker_attr=linker_attr)
        belt = get_outer_belt(G_apical)

        #initialize some things for the callback
        invagination = SG.arcs_and_pit(G, belt)
        # invagination = SG.arcs_with_intercalation(G, belt)
        # viewer = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)
        t_last = 0 
        t_plot = 1



        #create integrator
        integrate = vertex_integrator(G, G_apical, pre_callback=invagination, player=True, maxwell=True)
        #integrate
        try:
            integrate(0.5,2000, save_rate=1, save_pattern=f'data/testing/viscoelastic_tau_{tau}_*.pickle')
        except:
            print(f'failed to integrate tau={tau}')
            pass


if __name__ == '__main__':
    pool = Pool()
    pool.map( main, (600, 60, 6))

    print('done')
