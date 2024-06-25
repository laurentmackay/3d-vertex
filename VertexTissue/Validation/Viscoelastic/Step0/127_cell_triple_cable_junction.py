import os

from matplotlib import colors

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

from VertexTissue.Analysis import parameter_sweep, parameter_sweep_analyzer

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc
import VertexTissue.globals as const

from VertexTissue.util import first, get_myosin_free_cell_edges,  inside_arc, shortest_edge_length_and_time, shortest_edge_network_and_time

from VertexTissue.Iterable import first_item, imin

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/SAC+_127/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC+_127/'

from VertexTissue.Dict import last_dict_value
from VertexTissue.Geometry import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
lvl=3
inter_edges = ((305, 248), (94,163), (69,8), (2,8))

def run(force, visco=False,  phi0=1.0, level=0, arcs=3, cable=True):

    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    belt = get_outer_belt(G_apical)
# 
    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    # squeeze = SG.just_belt(G, belt, t=0, strength=force)
    
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
                                    blacklist=True,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    if visco:
        pattern = f'visco_phi0={phi0}_{force}.pickle'  if cable else f'visco_no_cable_phi0={phi0}_{force}.pickle' 
    else:
        pattern = f'elastic_edges_{force}.pickle'  if cable else f'elastic_no_cable_{force}.pickle' 

    if level != 0:
        pattern = f'lvl_{level}_'+ pattern

    if arcs==1:
        pattern='outer_'+pattern
    
    if arcs==0:
        pattern='peripheral_'+pattern

    if arcs==3:
        pattern='triple_'+pattern

    pattern=base_path+pattern

    # pattern=None
    print(f'starting f={force}')
    integrate(20, 6000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-4,
            save_rate=100,
            verbose=False,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def shortest_length(d, e=inter_edges[lvl]):
    return min([ euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for G in d.values() ])

def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
    # times = np.reshape([e[1] for e in results.ravel()], results.shape)
    inter_len = const.l_intercalation/const.l_apical

    
    # plt.plot(forces, lens[:,1],label='Elastic')
    # plt.plot(forces, lens[:,0],label='Elastic + Belt')
    # plt.plot(forces, lens[:,5],label='$\phi_0=0.8$ + Belt')
    # plt.plot(forces, lens[:,4],label='$\phi_0=0.6$ + Belt')
    # plt.plot(forces, lens[:,3],label='$\phi_0=0.45$ + Belt')
    # plt.plot(forces, lens[:,2],label='$\phi_0=0.3$ + Belt')
    phi = forces*const.myo_beta/const.l_apical
   
    for i in range(int(lens.shape[1]/2)):
        c=colors[i]
        if i>=1:
            prefix = f'$\phi_0$={phi0s[i-1]}'
            i=i+1
            j=i+len(phi0s)
        else:
            prefix='elastic'
            j=i+1

            
        plt.plot(phi, lens[:,i], color=c, label=prefix+' (cable)')
        plt.plot(phi, lens[:,j], linestyle='--', color=c)

    plt.xlabel(r'$-\phi^{\rm c}$', fontsize=16)
    plt.ylim((0,1))
    plt.ylabel(r'$\chi_{min}(\phi^{\rm c}| \phi_0 )$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'triple_belt_lvl_{lvl}_contraction_127.pdf',dpi=200)
    plt.show()

    thresh=0.03
    
    psi_elastic=phi[first(lens[:,1]<thresh)][0]
    # phi_elastic=phi[first(lens[:,1]<thresh)][0]
    psi_cable = np.array([phi[first(lens[:,i+2]<thresh)][0] for i, _ in enumerate(phi0s)])
    psi_no_cable = np.array([phi[first(lens[:,i+2+len(phi0s)]<thresh)][0] for i, _ in enumerate(phi0s)])


    alpha_VE = (psi_elastic-psi_no_cable)/psi_elastic
    alpha_cable = (psi_no_cable-psi_cable)/psi_elastic

    plt.figure().clear()


    plt.plot(phi0s, alpha_VE, label=r'$\alpha_{\rm VE}$')
    plt.plot(phi0s, alpha_cable, label=r'$\alpha_{\rm cable}$')
    plt.plot(phi0s,alpha_cable+alpha_VE, label=r'$\alpha_{\rm tot}$')
    
    plt.xlabel('$\phi_0$', fontsize=16)
    plt.ylabel(r'$\alpha$', fontsize=16)

    np.save(f'viscoelastic_cable_advantange_triple_127_level_{lvl}.npy', alpha_cable )
    np.save(f'viscoelastic_advantange_triple_127_level_{lvl}.npy', alpha_VE )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'viscoelastic_advantange_triple_127_level_{lvl}.pdf',dpi=200)
    plt.show()
    
    
    
def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0, **kw)

def visco_no_cable_runner(phi0, **kw):
    return lambda f: run(f, visco=True, cable=False, phi0=phi0, **kw)
# G_test, _ = tissue_3d( hex=7,  basal=True)
# belt = get_outer_belt(_)
# edges = get_myosin_free_cell_edges(G_test)
# nodes = np.unique(np.array([e for e in edges]).ravel())
# smth = [not inside_arc(n, inner_arc, G_test) for n in nodes]
# # inner_most = [inside_arc(n,outer_arc,G_test) for n in nodes]
# # smth = [inside_arc(n,outer_arc,G_test)  for n in nodes]
# bad_nodes = nodes[smth].ravel()

# def foo(d):
#     _, bar , min_len= shortest_edge_network_and_time(d,excluded_nodes=bad_nodes, return_length=True)
#     return min_len, bar

phi0s=[ .3,  .4, .5,  .6, .7, .8, .9]
colors=['k','r','g','b','y','m','c','orange']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(0,800,60)
    
    # forces=np.linspace(0,850,60)



    visco_funcs=[ *[visco_runner(phi, level=lvl) for phi in phi0s],
                    *[visco_no_cable_runner(phi, level=lvl) for phi in phi0s]]

    prefix='triple_'
    

    elastic_func=run
    funcs=[lambda f: run(f, cable=True, level=lvl), lambda f: run(f, cable=False, level=lvl), *visco_funcs]
    # if lvl==0:
    #     savepaths = [
    #         [base_path+f'{prefix}elastic_edges_{force}.pickle',
    #         *[base_path+f'{prefix}visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
    #         ] for force in forces
    #     ]
    # else:
    if lvl>0:
        lvl_str = f'lvl_{lvl}_'
    else:
        lvl_str =''

    savepaths = [
    [base_path+f'{prefix}{lvl_str}elastic_edges_{force}.pickle',
        base_path+f'{prefix}{lvl_str}elastic_no_cable_{force}.pickle',
    *[base_path+f'{prefix}{lvl_str}visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
    *[base_path+f'{prefix}{lvl_str}visco_no_cable_phi0={phi0}_{force}.pickle' for phi0 in phi0s]

    ] for force in forces
    ]
    if viewable:
        results = parameter_sweep(forces, funcs,  
                                  pre_process=shortest_length,
                                  savepaths=savepaths,
                                  overwrite=False,
                                  inpaint=np.nan,
                                  cache=False)
                                  
        plot_results(forces, results)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(forces[-2], visco=True, phi0=0.95)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

