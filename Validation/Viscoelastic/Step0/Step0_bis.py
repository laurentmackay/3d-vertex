import os

from matplotlib import colors
import VertexTissue
from VertexTissue.Memoization import function_call_savepath

from VertexTissue.visco_funcs import crumple
print('this is some basic output')

from doctest import FAIL_FAST
from itertools import product
from pathos.pools import ProcessPool as Pool
import numpy as np


from VertexTissue.vertex_3d import monolayer_integrator
from VertexTissue.Tissue import T1_minimal, T1_shell, tissue_3d, get_outer_belt
import VertexTissue.SG as SG
import VertexTissue.T1 as T1
from VertexTissue.Sweep import sweep

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc
import VertexTissue.globals as const

from VertexTissue.util import first, get_myosin_free_cell_edges,  inside_arc, shortest_edge_length_and_time, shortest_edge_network_and_time

from VertexTissue.Iterable import first_item, imin

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

from VertexTissue.Dict import last_dict_value
from VertexTissue.Geometry import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)

inter_edges = ((305, 248), (94,163), (69,8), (2,8))

def run(force,  phi0=1.0, level=0, arcs=3, visco=True, cable=True):

    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    belt = get_outer_belt(G_apical)


    #initialize some things for the callback

    
    if arcs==0:
        arc_list=(belt,)
    if arcs==1:
        arc_list=(outer_arc,)
    elif arcs==2:
        arc_list=(outer_arc, belt)
    elif arcs==3:
        arc_list=(outer_arc, inner_arc, belt)

    squeeze = SG.arcs_and_intercalation(G, arc_list, t_belt=0, inter_edges=(inter_edges[level],),  t_intercalate=0, intercalation_strength=force,
                                        arc_strength = belt_strength if cable else 0.0)


    # const.press_alpha/=4
    if not visco:
        kw={}
    else:
        kw={'rest_length_func':crumple(phi0=phi0)}

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
                                    blacklist=True, length_abs_tol=1e-2,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')

    # pattern=None
    print(f'starting f={force}')
    ec=0.2
    k_eff = (phi0-ec)/(1-ec)

    dt_min = 1e-4 if not cable else 1e-5

    integrate(5, 8000, 
            dt_init = 1e-4,
            adaptive=True,
            dt_min=dt_min*k_eff,
            save_rate=100,
            verbose=False,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def shortest_length(d, e=None, level=None, **kw):
    if e is None:
        e=inter_edges[level]
    return min([ euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for G in d.values() ])

def plot_results_no_cable(forces,results, level=None):
    if not viewable:
        return

    N=results.shape[-1]    
    colors = plt.cm.nipy_spectral(np.linspace(0,.85,N))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors )



    lens = results/const.l_apical
    # times = np.reshape([e[1] for e in results.ravel()], results.shape)
    inter_len = const.l_intercalation/const.l_apical

    phi = forces*const.myo_beta/const.l_apical
   
    for i in range(int(lens.shape[-1])):
        # c=colors[i]

        prefix = f'$\phi_0$={phi0s[i]}'
        # i=i+1
        # j=i+len(phi0s)


            
        plt.plot(phi, lens[:,0,i],  label=prefix)
        # plt.plot(phi, lens[:,j], linestyle='--', color=c)

    plt.xlabel(r'$-\phi^{\rm j}$', fontsize=16)
    plt.ylim((0,1))
    plt.ylabel(r'$\frac{\ell^*}{L_0}$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'triple_belt_lvl_{level}_cable_contraction_127.pdf',dpi=200)
    plt.show()

    thresh=0.03
    
    # psi_elastic=phi[first(lens[:,1]<thresh)][0]
    # # phi_elastic=phi[first(lens[:,1]<thresh)][0]
    # psi_cable = np.array([phi[first(lens[:,i+2]<thresh)][0] for i, _ in enumerate(phi0s)])
    # psi_no_cable = np.array([phi[first(lens[:,i+2+len(phi0s)]<thresh)][0] for i, _ in enumerate(phi0s)])


    # alpha_VE = (psi_elastic-psi_no_cable)/psi_elastic
    # alpha_cable = (psi_no_cable-psi_cable)/psi_elastic

    # plt.figure().clear()


    # plt.plot(phi0s, alpha_VE, label=r'$\alpha_{\rm VE}$')
    # plt.plot(phi0s, alpha_cable, label=r'$\alpha_{\rm cable}$')
    # plt.plot(phi0s,alpha_cable+alpha_VE, label=r'$\alpha_{\rm tot}$')
    
    # plt.xlabel('$\phi_0$', fontsize=16)
    # plt.ylabel(r'$\alpha$', fontsize=16)

    # # np.save(f'viscoelastic_cable_advantange_triple_127_level_{lvl}.npy', alpha_cable )
    # # np.save(f'viscoelastic_advantange_triple_127_level_{lvl}.npy', alpha_VE )

    # plt.legend()
    # plt.tight_layout()
    # # plt.savefig(f'viscoelastic_advantange_triple_127_level_{lvl}.pdf',dpi=200)
    # plt.show()
    
    
    

phi0s=list(reversed([ .22, .25, .3,  .4, .5,  .6, .7, .8, .9, .95, 1.0]))
# colors=['k','r','g','b','y','m','c','orange']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(0, 600, 80)
    
    # forces=np.linspace(0,850,60)



    kws={'cable':[False,True], 'level':[0,2,3],  'phi0':phi0s}
    
    kws_no_cable_0={'cable':[False], 'level':0,  'phi0':phi0s}
    kws_no_cable_1={'cable':[False], 'level':1, 'phi0':phi0s}
    kws_no_cable_2={'cable':[False], 'level':2,  'phi0':phi0s}
    kws_no_cable_3={'cable':[False], 'level':3,  'phi0':phi0s}

    kws_0={'cable':[True], 'level':0,  'phi0':phi0s}
    kws_1={'cable':[True], 'level':1, 'phi0':phi0s}
    kws_2={'cable':[True], 'level':2,  'phi0':phi0s}
    kws_3={'cable':[True], 'level':3,  'phi0':phi0s}


    kws=kws_1
    
    results = sweep(forces, run,  kw = kws,
        pre_process=shortest_length,
        savepath_prefix=base_path,
        overwrite=False,
        cache=True,
        pass_kw=True,
        inpaint=np.nan if viewable else None, refresh=True)

    if viewable:
        plot_results_no_cable(forces,results, level=1)


    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

