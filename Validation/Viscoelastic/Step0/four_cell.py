from doctest import FAIL_FAST
from itertools import product
from pathos.pools import ProcessPool as Pool
import numpy as np
import matplotlib.pyplot as plt

from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge
import VertexTissue.globals as const


from VertexTissue.util import last_item
from VertexTissue.funcs import euclidean_distance


try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/T1/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/T1/'


def l_ss(phi, ec=0.2, phi0=1.0):
    if phi<ec:
        return 1-phi
    elif phi<=phi0:
        return (1-ec)+(1-ec)*(ec-phi)/(phi0-ec)
    else:
        return 0.0




def run(force, visco=False, phi0=1.0, ndim=3):

    #
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=True)
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    intercalation = T1.simple_T1(G, strength=force)


    def l_rest(ell,e,ec=.2):

            L0=G[e[0]][e[1]]['l_rest']

            if ell<(1-ec)*L0:
                    return (1-phi0)*ell/(1-ec)+ phi0*L0
            else:
                    return L0

   
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
                                    pre_callback=intercalation, 
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,  
                                    length_rel_tol=0.005,
                                    player=False, viewer=viewable, minimal=False, blacklist=True, **kw)
    #integrates
    # try:
    print(f'starting f={force}')
    pattern = f'visco_phi0={phi0}_{force}.pickle' if visco else f'elastic_edges_{force}.pickle'
    pattern=base_path+pattern
    integrate(1, 2000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-5,
            save_rate=5,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def visco_runner(phi0):
    return lambda f: run(f, visco=True, phi0=phi0)

def final_length(d):
    G = last_item(d)
    # edge_view(G)
    a = G.nodes[1]['pos']
    b = G.nodes[11]['pos']
    return euclidean_distance(a,b)

def shortest_length(d):
    lens = []
    for G in d.values():
        a = G.nodes[1]['pos']
        b = G.nodes[11]['pos']
        lens.append(euclidean_distance(a,b))

    # edge_view(G)

    return min(lens)
colors=['r','g','b']
def plot_results(forces,results):

    inter_len = const.l_intercalation/const.l_apical
    
    lens = results/const.l_apical


    phi = forces*const.myo_beta/const.l_apical
    phi_fine = np.linspace(phi[0],phi[-1],200)


    for i, phi0 in enumerate(phi0s):
        plt.plot(phi, lens[:,1+i],linestyle='-',color=colors[i],label=f'$\phi_0={phi0}$')

    plt.plot(phi, lens[:,0],'-k',label='elastic ($\phi_0=1$)')

    plt.xlim((0,2))
    plt.plot([phi[0], phi[-1]],[inter_len, inter_len],':k')


    plt.xlabel(r'$-\phi^{\rm j}$', fontsize=16)
    plt.ylabel(r'$\chi(\phi^{\rm j}, 0| \phi_0 )$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('T1_VE.pdf',dpi=200)
    plt.show()

if __name__ == '__main__':
    forces=np.linspace(0, 800, 60)
    phi0s=[0.3, .6, .8]
    # visco_conds=product((True,), forces, phi0)
    # elastic_conds=product((False,), forces)
    visco_funcs=[ visco_runner(phi) for phi in phi0s]



    elastic_func=run
    funcs=[elastic_func, *visco_funcs]
    savepaths = [
        [base_path+f'elastic_edges_{force}.pickle',
        *[base_path+f'visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s]] for force in forces
    ]
    if viewable:
        results=parameter_sweep(forces, funcs,  savepaths=savepaths, overwrite=False, pre_process=shortest_length, inpaint=np.nan)
        plot_results(forces, results)
        # run(0.001,False)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # Pool(nodes=6).map( run, taus)
    # run(150,False)
    print('done')
