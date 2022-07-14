import numpy as np
import time

from pathos.pools import ProcessPool as Pool

from VertexTissue.funcs import unit_vector_2D
from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import linear_grid
import VertexTissue.globals as const


# const.mu_apical=1
fmags=np.linspace(0,1)

def run(fmag):

    N=1

    G = linear_grid( N, edge_attrs={'l_rest':1.0,'l_rest_0':1.0,'tension':0.0,'tau':360,'myosin':0.0}, h=1.0)





    forced_points=(0,1)

    

    def forcing(t,force_dict):

        if t<20000:
            pos_a = G.nodes[0]['pos']
            pos_b = G.nodes[1]['pos']
            force_vec = -fmag*unit_vector_2D(pos_a, pos_b)
            forces=[-force_vec,  force_vec]   

            for p,f in zip(forced_points, forces):
                force_dict[p] += f



    #create integrator
    integrate = monolayer_integrator(G, G, pre_callback=forcing, ndim=2, viewer=False , maxwell=True, save_rate=-1, tension=True, tension_remodel=True, SLS=True)
    #integrate
    integrate(1, 30000, adaptive=True, save_pattern='data/tension/rod_relax_'+f'force_{fmag}_*.pickle')

    print(f'Done: f={fmag}')



if __name__ == '__main__':

    
    pool = Pool(nodes=6)

 

    pool.map( run, fmags)
    # run(0.5)

    
    print('Done ALL')
    # initialize the tissue
