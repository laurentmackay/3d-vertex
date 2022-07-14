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
    G, G_apical = tissue_3d( gen_centers=T1_minimal,  basal=(ndim==3))
    
    belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    intercalation = T1.simple_T1(G, strength=force)
    squeeze = SG.just_belt(G, belt, t=0, strength=force)

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


    const.l_mvmt=0.001
    if ndim==2:
        const.press_alpha*=40
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, blacklist=True, ndim=ndim, **kw)
    #integrates
    # try:
    print(f'starting f={force}')
    if ndim==3:
        pattern = f'visco_cable_phi0={phi0}_{force}.pickle' if visco else f'elastic_cable_{force}.pickle'
    elif ndim==2:
        pattern = f'visco_cable_2D_phi0={phi0}_{force}.pickle' if visco else f'elastic_cable_2D_{force}.pickle'

    pattern=base_path+pattern
    integrate(2, 2000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-2,
            save_rate=50,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0,**kw)

def final_length(d):
    G = last_item(d)
    # edge_view(G)
    a = G.nodes[1]['pos']
    b = G.nodes[11]['pos']
    return euclidean_distance(a,b)
edges=((1,11),(10,11))
def shortest_length(d):
    lens = []
    for G in d.values():
        # a = G.nodes[1]['pos']
        # b = G.nodes[11]['pos']
        # c = G.nodes[10]['pos']
        lens.append(min(*[euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for e in edges]))

    # edge_view(G)

    return min(lens)

def plot_results(forces,results):

    inter_len = const.l_intercalation/const.l_apical
    
    lens = results/const.l_apical


    phi = forces*const.myo_beta/const.l_apical
    phi_fine = np.linspace(phi[0],phi[-1],200)

    

    plt.plot(phi, lens[:,1],'-r',label='$\phi_0=0.3$  (3D)')
    plt.plot(phi, lens[:,5],'--r')

    plt.plot(phi, lens[:,2],'-g',label='$\phi_0=0.6$  (3D)')
    plt.plot(phi, lens[:,6],'--g')

    plt.plot(phi, lens[:,3],'-b',label='$\phi_0=0.8$  (3D)')
    plt.plot(phi, lens[:,7],'--b')

    plt.plot(phi, lens[:,0],'-k',label='elastic ($\phi_0=1$)')
    plt.plot(phi, lens[:,4],'--k')
    # plt.plot(phi_fine, [l_ss(f,phi0=0.8) for f in phi_fine],'--g',label='single edge')


    # plt.plot(phi_fine, [l_ss(f,phi0=0.8) for f in phi_fine],'--g',label='single edge')


    # plt.plot(phi_fine, [l_ss(f,phi0=0.3) for f in phi_fine],'--r',label='single edge')

    

    plt.plot([phi[0], phi[-1]],[inter_len, inter_len],':k')


    plt.xlabel(r'$-\phi^{\rm c}$', fontsize=16)
    plt.ylabel(r'$\chi_{min}(\phi^{\rm c}| \phi_0 )$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('T1_cable_VE.pdf',dpi=200)
    plt.show()

if __name__ == '__main__':
    forces=np.linspace(0, 1200, 40)
    phi0s=[0.3, 0.6, .8]
    # visco_conds=product((True,), forces, phi0)
    # elastic_conds=product((False,), forces)
    visco_funcs=[ visco_runner(phi) for phi in phi0s]
    visco_funcs_2D=[ visco_runner(phi, ndim=2) for phi in phi0s]


    elastic_func=run
    elastic_func_2D=lambda f: run(f,ndim=2)
    funcs=[elastic_func, *visco_funcs, elastic_func_2D, *visco_funcs_2D ]


    savepaths = [
        [base_path+f'elastic_cable_{force}.pickle',
        *[base_path+f'visco_cable_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
        base_path+f'elastic_cable_2D_{force}.pickle',
        *[base_path+f'visco_cable_2D_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
        ] for force in forces
    ]
    if viewable:
        parameter_sweep_analyzer(forces, funcs, plot_results, savepaths=savepaths, overwrite=False, pre_process=shortest_length, inpaint=None)
        # run(0.00001, visco=True, phi0=0.3, ndim=3)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(0.00001, visco=True, phi0=0.3, ndim=3)
    # 
    print('done')
