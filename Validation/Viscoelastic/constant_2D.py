import numpy as np

from VertexTissue.funcs import unit_vector_2D
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d

forces = (1, 2.5, -1, -2.5)
times = (800, 800, 320, 115)


if __name__ == '__main__':

    

    # initialize the tissue
    N=1
    M=1
    for f, t_final in zip(forces, times):
        G = square_grid_2d( N, M)
        l1 = list(range(M+1))
        l2 = list(range(len(G)-M-1, len(G)))
        forced_points = [ *l1 , *l2]
        pos_a = G.nodes[forced_points[0]]['pos']
        pos_b = G.nodes[forced_points[-(M+1)]]['pos']

        force_vec = f*unit_vector_2D(pos_a, pos_b)

        for a,b in zip(l1,l2):
            G[a][b]['tau'] = 60

        # forced_points=[range(N+1)]

        forces=[-force_vec,  -force_vec,   force_vec,  force_vec]   

        def forcing(t,force_dict):
            for p,f in zip(forced_points, forces):
                force_dict[p] += f



        #create integrator
        integrate = monolayer_integrator(G, G, pre_callback=forcing, ndim=2, save_rate=1, maxwell=True)
        #integrate
        integrate(0.1, t_final, save_pattern='data/viscoelastic/'+f'force_{f}_*.pickle')

    print('Done')