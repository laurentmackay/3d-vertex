import os
from pathos.pools import ProcessPool as Pool
import numpy as np
import matplotlib.pyplot as plt

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, T1_shell, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge
import VertexTissue.globals as const

from VertexTissue.PyQtViz import edge_view
from VertexTissue.util import last_dict_value
from VertexTissue.Geometry import euclidean_distance


dt=0.1
taus = np.logspace(6,1,5)

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

def run(force, visco=False, phi0=1.0):

    #
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=False)
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 

    const.press_alpha*=100

    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    squeeze = SG.just_belt(G, belt, t=0, strength=force)

    def l_rest(ell,e,ec=.2):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

   
    if not visco:
        kw={'ndim':2}
    else:
        kw={'rest_length_func':l_rest, 'ndim':2}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    const.l_intercalation=0.1
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,  
                                    length_rel_tol=0.01,
                                    player=False, viewer=True, minimal=False, **kw)
    #integrates
    # try:
    # print(belt)
    print(f'starting f={force}')
    pattern = f'./data/SAC/four_cell_2D_visco_phi0={phi0}_{force}.pickle' if visco else f'./data/SAC/four_cell_2D_elastic_edges_{force}.pickle'

    # decrease_nice()

    integrate(1, 2000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-4,
            save_rate=10,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def visco_runner(phi0):
    return lambda f: run(f, visco=True, phi0=phi0)

def final_length(d):
    G = last_dict_value(d)
    # edge_view(G)
    a = G.nodes[1]['pos']
    b = G.nodes[11]['pos']
    return euclidean_distance(a,b)


def plot_results(forces,results):
    
    lens = np.frompyfunc(final_length,1,1)(results)
    plt.plot(forces, lens/const.l_apical)
    plt.show()
    print('this is a print statement')

if __name__ == '__main__':
    forces=np.linspace(0, 2500, 10)
    phi0s=[0.3, ]
    # visco_conds=product((True,), forces, phi0)
    # elastic_conds=product((False,), forces)
    visco_funcs=[ visco_runner(phi) for phi in phi0s]



    elastic_func=run
    funcs=[elastic_func, *visco_funcs]
    savepaths = [
        [f'./data/SAC/four_cell_2D_elastic_edges_{force}.pickle',
        *[f'./data/SAC/four_cell_2D_visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s]] for force in forces
    ]

    # parameter_sweep_analyzer(forces, funcs, plot_results, savepaths=savepaths, overwrite=True)

    run(8000,True, phi0=.3)
    print('done')
