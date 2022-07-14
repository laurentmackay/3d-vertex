import os

from matplotlib import colors
print('this is some basic output')

from doctest import FAIL_FAST
from itertools import product
from pathos.pools import ProcessPool as Pool
import numpy as np


from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, T1_shell, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength
import VertexTissue.globals as const

from VertexTissue.util import first

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/14_junction/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/14_junction/'

from VertexTissue.util import last_item
from VertexTissue.funcs import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
def run(force, visco=False, cable=True, phi0=1.0, ndim=3):

    #
    G, G_apical = tissue_3d( gen_centers=T1_shell,  basal=(ndim==3))
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    squeeze = SG.just_intercalation(G, ((11,1),), t_intercalate=0, intercalation_strength=force)

    def l_rest(ell,e,ec=.2):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

    # const.press_alpha/=4
    if not visco:
        kw={}
    else:
        kw={'rest_length_func':l_rest}

    done=False
    def terminate(*args):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    const.l_intercalation=0.1
    const.l_mvmt=0.001
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,  
                                    blacklist=True,
                                    player=False, viewer={'button_callback':terminate,'nodeLabels':None} if viewable else False, minimal=False, ndim=ndim, **kw)
    #integrates
    # try:
    # print(belt)
    if visco:
        pattern = f'visco_phi0={phi0}_{force}.pickle' if ndim==3 else f'visco_2D_phi0={phi0}_{force}.pickle' 
    else:
        pattern = f'elastic_edges_{force}.pickle' if ndim==3 else f'elastic_edges_2D_{force}.pickle'

    pattern=base_path+pattern

    # pattern=None
    print(f'starting f={force}')
    integrate(2, 2000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-2 if ndim==3 else 1e-1,
            save_rate=50,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass



def final_length(d):
    G = last_item(d)
    # edge_view(G)
    a = G.nodes[8]['pos']
    b = G.nodes[41]['pos']
    return euclidean_distance(a,b)


edges=((1,11),(9,47),)
def shortest_length(d):
    lens = []
    for G in d.values():
        lens.append(min(*[euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for e in edges]))
    
    # edge_view(G)

    return min(lens)


def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
    inter_len = const.l_intercalation/const.l_apical


    phi = forces*const.myo_beta/const.l_apical
    plt.plot(phi, lens[:,0],linestyle='-',color='k', label='elastic ($\phi_0=1$)')
    plt.plot(phi, lens[:,1],linestyle='--',color='k')
    
    for i, fi, c in zip(range(len(phi0s)),phi0s,colors):
        plt.plot(phi, lens[:,2+i],linestyle='-',color=c, label=f'$\phi_0={fi}$ (3D)')
        plt.plot(phi, lens[:,2+i+len(phi0s)],linestyle='--',color=c)


    plt.plot([phi[0], phi[-1]],[inter_len, inter_len],':k')
    plt.xlabel(r'$-\phi^{\rm j}$', fontsize=16)
    plt.ylabel(r'$\chi(\phi^{\rm j}, 0| \phi_0 )$', fontsize=16)
    plt.xlim([0,2])

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('fourteen_cell_junction_contraction.pdf',dpi=200)
    plt.show()
    
   

def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0, **kw)



phi0s=[0.3,   0.6, .8]
colors=['r','g','b','y','m','c','orange']
def main():
    forces=np.linspace(0, 1200, 40)
    
    # visco_conds=product((True,), forces, phi0)
    # elastic_conds=product((False,), forces)
    visco_funcs=[ *[visco_runner(phi) for phi in phi0s],
                    *[visco_runner(phi, ndim=2) for phi in phi0s]]



    elastic_func=run
    funcs=[elastic_func, lambda f: run(f, ndim=2), *visco_funcs]
    savepaths = [
        [base_path+f'elastic_edges_{force}.pickle',
        base_path+f'elastic_edges_2D_{force}.pickle',
        *[base_path+f'visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
        *[base_path+f'visco_2D_phi0={phi0}_{force}.pickle' for phi0 in phi0s]
         ] for force in forces
    ]

    if viewable:
        results=parameter_sweep(forces, funcs, pre_process=shortest_length, savepaths=savepaths, overwrite=False, cache=True)
        plot_results(forces, results)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(0.001, visco=True, phi0=0.3, ndim=3)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

