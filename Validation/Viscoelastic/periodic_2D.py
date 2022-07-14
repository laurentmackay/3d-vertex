import numpy as np

from VertexTissue.funcs import unit_vector_2D
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import square_grid_2d

f = 2.5
omega = 1

if __name__ == '__main__':

    

    # initialize the tissue
    N=1
    M=1

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
            force_dict[p] += f*np.sin(t*omega)



    #create integrator
    integrate = monolayer_integrator(G, G, pre_callback=forcing, ndim=2, save_rate=0.1, maxwell=True)
    #integrate
    integrate(0.01, 200, save_pattern='data/viscoelastic/'+f'periodic_transient_force_{f}_freq_{omega}_*.pickle')

    print('Done')