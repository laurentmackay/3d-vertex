from asyncio.subprocess import Process
from pathos.pools import ProcessPool as Pool
import numpy as np

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1
from VertexTissue.funcs import unit_vector, unit_vector_2D

from VertexTissue.globals import default_ab_linker, default_edge, basal_offset, myo_beta, belt_strength

from VertexTissue.PyQtViz import edge_view

edge_attr = default_edge.copy()
linker_attr = default_ab_linker.copy()
spoke_attr = default_edge.copy()
dt=0.1
tau=60
def main(fmag):

    
    #setup the desired rod attributes
    # spoke_attr['tau'] = tau
    # linker_attr['tau'] = tau
    # edge_attr['tau'] = tau
    
    # initialize the tissue
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)
    # G, G_apical = tissue_3d(  basal=True, cell_edge_attr=edge_attr, linker_attr=linker_attr, spoke_attr=spoke_attr)

    # edge_view(G, exec=True)


    #initialize some things for the callback
    intercalation = T1.simple_T1(G)
    # invagination = SG.arcs_and_pit(G,belt)

    keep_forcing=True
    def stop_forcing(*args):
        nonlocal keep_forcing
        keep_forcing=False

    compressed = (16,15,9,3,19,18,13,5)
    compressed = ((16,14),(15,14),(9,7),(10,7),(3,0),(2,0),(19,17),(18,17),(13,7),(12,7),(5,0),(6,0))
    compression_pairs = ((16,18),(15,19),(2,6),(5,3),(13,9),(12,10))
    origin = np.array((0.,0.,0.))
    compression_pairs = ((16,18),(15,19))

    forced_points = (11,1)
    offsets=(0, basal_offset)
    corners=(4,8,12,10,6,2)
    ndim=3
    # G[1][11]['tau']=6000
    G[1][11]['myosin']=belt_strength*1.25
    # G[1+basal_offset][11+basal_offset]['myosin']=belt_strength*1.25
    def forcing(t,force_dict):

        pos_a = G.nodes[11]['pos']
        pos_b = G.nodes[1]['pos']

        force_vec = belt_strength*myo_beta*unit_vector(pos_a, pos_b)
        forces=[force_vec, -force_vec]

        # if keep_forcing:
        #     for p,f in zip(forced_points, forces):
        #         force_dict[p] += f[:ndim]
        if t>0:
            for _ in compression_pairs:
                i,j = _
                for o in offsets:
                    b = G.nodes[i+o]['pos']
                    a =  G.nodes[j+o]['pos']
                    v = unit_vector(b, a)
                    force_dict[i+o] += fmag*v[:ndim]
                    force_dict[j+o] += -fmag*v[:ndim]
        
        # for i in corners:
        #     for o in offsets:
        #         force_dict[i+o][0] = 0

        

    #create integrator
    integrate = monolayer_integrator(G, G_apical, pre_callback=forcing, intercalation_callback=stop_forcing, player=False, viewer={'draw_axes':True},
                                  maxwell=True, minimal=False, 
                                  blacklist=True, ndim=ndim)
    #integrates
    # try:
    integrate(dt, 400000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=0,
            save_rate=0.5,
            length_prec=0.01,
            save_pattern='./data/T1/elastic_*.pickle', ndim=ndim)
    # except:
    #     print(f'failed to integrate tau={tau}')
    #     pass


if __name__ == '__main__':
    main(1.3)

    print('done')
