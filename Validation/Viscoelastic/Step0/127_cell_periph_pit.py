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

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc, pit_centers
import VertexTissue.globals as const

from VertexTissue.util import first, imin, shortest_edge_length_and_time, shortest_edge_network_and_time

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/SAC_pit/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC_pit/'

from VertexTissue.util import last_dict_value
from VertexTissue.Geometry import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
lvl=0
inter_edges = ((174,163),(83,94), (82,69), (2,8))
arcs=0

def run(force, visco=False,  phi0=1.0, level=0, arcs=0):

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

    squeeze = SG.arcs_and_pit(G, arcs=arc_list, t_pit=0.0, t_arcs=0, arc_strength=force)

    
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
                                    player=False, viewer={'button_callback':terminate,'nodeLabels':None} if viewable else False, minimal=False, **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    if visco:
        pattern = f'visco_phi0={phi0}_{force}.pickle'
    else:
        pattern = f'elastic_edges_{force}.pickle' 

    if level != 0:
        pattern = f'lvl_{level}_'+ pattern

    if arcs==1:
        pattern='outer_'+pattern
    
    if arcs==0:
        pattern='peripheral_'+pattern

    if arcs==3:
        pattern='triple_'+pattern

    pattern=base_path + pattern

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



def final_length(d):
    G = last_dict_value(d)
    # edge_view(G)
    a = G.nodes[174]['pos']
    b = G.nodes[163]['pos']
    return euclidean_distance(a,b)

def shortest_length(d):
    min_lens = []
    G=d[list(d.keys())[1]]
    centers = G.graph['centers']
    edges = [e for e in G.edges if (e[0] not in centers ) and (e[1] not in centers) and G[e[0]][e[1]]['myosin']==0 ]
    for G in d.values():
       
        lens = [ euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos']) for a,b in edges]
        min_lens.append(min(lens))

    
    # edge_view(G)

    return min(min_lens)

def shortest_length_time(d):
    min_lens = []

    times=list(d.keys())
    G=d[times[1]]
    centers = G.graph['centers']
    edges = [e for e in G.edges if (e[0] not in centers ) and (e[1] not in centers) and G[e[0]][e[1]]['myosin']==0 ]
    for t in times:
        G=d[t]
        lens = [ euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos']) for a,b in edges]
        min_lens.append(min(lens))

    
    # edge_view(G)

    return times[imin(min_lens)]


def plot_results(forces,results):
    if not viewable:
        return
    
    lens = np.reshape([e[0] for e in results.ravel()], results.shape)/const.l_apical
    times = np.reshape([e[1] for e in results.ravel()], results.shape)
    inter_len = const.l_intercalation/const.l_apical

    
    # plt.plot(forces, lens[:,1],label='Elastic')
    # plt.plot(forces, lens[:,0],label='Elastic + Belt')
    # plt.plot(forces, lens[:,5],label='$\phi_0=0.8$ + Belt')
    # plt.plot(forces, lens[:,4],label='$\phi_0=0.6$ + Belt')
    # plt.plot(forces, lens[:,3],label='$\phi_0=0.45$ + Belt')
    # plt.plot(forces, lens[:,2],label='$\phi_0=0.3$ + Belt')
    phi = forces*const.myo_beta/const.l_apical
   
    for i in range(int(lens.shape[1])):
        c=colors[i]
        if i<lens.shape[1]-1:
            prefix = f'$\phi_0$={phi0s[i]}'
            i=i+1
            # j=i+len(phi0s)
        else:
            prefix='elastic'
            i=0

            
        plt.plot(phi,lens[:,i],color=c,label=prefix)
        # plt.plot(phi,lens[:,j],linestyle='--',color=c,label=prefix+' (no cable)')

    plt.xlabel(r'$-\phi^{\rm c}$', fontsize=16)
    plt.ylim((0,1))
    plt.ylabel(r'$\chi_{min}(\phi^{\rm c}| \phi_0 )$', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'pit_belt_contraction_127.pdf',dpi=200)
    plt.show()

    plt.figure().clear()

    for i in range(int(lens.shape[1])):
        c=colors[i]
        if i<lens.shape[1]-1:
            prefix = f'$\phi_0$={phi0s[i]}'
            i=i+1
            # j=i+len(phi0s)
        else:
            prefix='elastic'
            i=0

            
        plt.plot(phi,times[:,i],color=c,label=prefix)
        # plt.plot(phi,lens[:,j],linestyle='--',color=c,label=prefix+' (no cable)')z

    
    plt.xlabel('$-\phi$ (cable)', fontsize=16)
    plt.ylabel('Shortest Edge Time', fontsize=16)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'pit_belt_contraction_time_127.pdf',dpi=200)
    plt.show()
    
    
    
def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0, **kw)

def visco_no_cable_runner(phi0, **kw):
    return lambda f: run(f, visco=True, cable=False, phi0=phi0, **kw)

phi0s=[ .3,  .4, .5,  0.6, .7, .8, .9]
colors=['r','g','b','y','m','c','orange','k']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces =np.array( [*np.linspace(0,850,60), *np.linspace(850,1200,21)[1:], *np.linspace(1200,1500,21)[1:]])
    
    # forces=np.linspace(0,850,60)



    visco_funcs=[ *[visco_runner(phi, level=lvl) for phi in phi0s],
                    ]

    if arcs==1:
        prefix='outer_'
    
    if arcs==0:
        prefix='peripheral_'

    if arcs==2:
        prefix='peripheral_outer_'

    if arcs==3:
        prefix='triple_'
    

    funcs=[lambda f: run(f, level=lvl), *visco_funcs]
    if lvl==0:
        savepaths = [
            [base_path+f'{prefix}elastic_edges_{force}.pickle',
            *[base_path+f'{prefix}visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
            ] for force in forces
        ]
    else:
        savepaths = [
            [base_path+f'lvl_{lvl}_elastic_edges_{force}.pickle',
           
            *[base_path+f'lvl_{lvl}_visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
           
            ] for force in forces
        ]


    if viewable:
        
        G, G_apical = tissue_3d( hex=7,  basal=True)
        nhbrs = [n  for c in pit_centers for n in list(G.neighbors(c))]
        pre_process=lambda d: shortest_edge_length_and_time(d, excluded_nodes=nhbrs)
        results = parameter_sweep(forces, funcs,  
                                  pre_process=pre_process,
                                  savepaths=savepaths,
                                  overwrite=False,
                                  inpaint=(np.nan, np.nan),
                                  cache=True)
                                  
        plot_results(forces, results)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(forces[-2], visco=True, phi0=0.3)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

