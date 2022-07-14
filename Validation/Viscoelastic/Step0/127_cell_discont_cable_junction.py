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

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc
import VertexTissue.globals as const

from VertexTissue.util import first

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/SAC+127/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/SAC+127/'

from VertexTissue.util import last_item
from VertexTissue.funcs import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
lvl=1
inter_edges = ((174,163),(83,94), (82,69), (2,8))
def run(force, visco=False, cable=True, phi0=1.0, level=0, arcs=1):

    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    # belt = get_outer_belt(G_apical)

    # if visco:
    #     const.mu_apical*=phi0 


    #initialize some things for the callback
    # intercalation = T1.simple_T1(G, strength=force)
    # squeeze = SG.just_belt(G, belt, t=0, strength=force)
    
    if arcs==1:
        arc_list=(outer_arc,)
    elif arcs==2:
        arc_list=(outer_arc, inner_arc)

    squeeze = SG.arcs_and_intercalation(G, arc_list, t_belt=0, inter_edges=(inter_edges[level],),  t_intercalate=0, intercalation_strength=force, arc_strength=belt_strength if cable else 0.0)
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
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)
    #integrates
    # try:
    # print(belt)
    if visco:
        pattern = f'visco_phi0={phi0}_{force}.pickle' if cable else f'visco_no_cable_phi0={phi0}_{force}.pickle' 
    else:
        pattern = f'elastic_edges_{force}.pickle' if cable else f'elastic_no_cable_{force}.pickle'

    if level != 0:
        pattern = f'lvl_{level}_'+ pattern

    pattern=base_path+pattern

    # pattern=None
    print(f'starting f={force}')
    integrate(2, 1000, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=1e-4,
            save_rate=50,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass



def final_length(d):
    G = last_item(d)
    # edge_view(G)
    a = G.nodes[174]['pos']
    b = G.nodes[163]['pos']
    return euclidean_distance(a,b)

def shortest_length(d):
    lens = []
    e=inter_edges[lvl]
    for G in d.values():
        a = G.nodes[e[0]]['pos']
        b = G.nodes[e[1]]['pos']
        lens.append(euclidean_distance(a,b))
    
    # edge_view(G)

    return min(lens)


def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
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
        else:
            prefix='elastic'
        if i>=1:
            i=i+1
            j=i+len(phi0s)
        else:
            j=i+1
        plt.plot(phi,lens[:,i],color=c,label=prefix+' (cable)')
        plt.plot(phi,lens[:,j],linestyle='--',color=c,label=prefix+' (no cable)')


    plt.xlabel('$-\phi$ (junctional)', fontsize=14)
    plt.ylabel('Radial Edge Length (norm.)', fontsize=14)
    # plt.plot(forces, lens[:,2],'-r',label='$\phi_0=0.3$')
    # plt.plot(forces, lens[:,6],'--r')
    # plt.plot(forces, lens[:,3],'-g',label='$\phi_0=0.45$')
    # plt.plot(forces, lens[:,7],'--g')
    # plt.plot(forces, lens[:,4],'-b',label='$\phi_0=0.6$')
    # plt.plot(forces, lens[:,8],'--b')
    # plt.plot(forces, lens[:,5],'-y',label='$\phi_0=0.8$')
    # plt.plot(forces, lens[:,9],'--y')

    plt.legend()
    plt.tight_layout()
    # plt.savefig('belt_junction_contraction.pdf',dpi=200)
    plt.show()
    
def visco_runner(phi0, **kw):
    return lambda f: run(f, visco=True, phi0=phi0, **kw)

def visco_no_cable_runner(phi0, **kw):
    return lambda f: run(f, visco=True, cable=False, phi0=phi0, **kw)

phi0s=[0.3,  .4, .5,  0.6, .7, .8]
colors=['r','g','b','y','m','c','orange']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    # forces = np.linspace(0,850,40)




    visco_funcs=[ *[visco_runner(phi, level=lvl) for phi in phi0s],
                    *[visco_no_cable_runner(phi, level=lvl) for phi in phi0s]]

    

    elastic_func=run
    funcs=[lambda f: run(f, cable=False, level=lvl), lambda f: run(f, cable=False, level=lvl), *visco_funcs]
    if lvl==0:
        savepaths = [
            [base_path+f'elastic_edges_{force}.pickle',
            base_path+f'elastic_no_cable_{force}.pickle',
            *[base_path+f'visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
            *[base_path+f'visco_no_cable_phi0={phi0}_{force}.pickle' for phi0 in phi0s]
            ] for force in forces
        ]
    else:
        savepaths = [
            [base_path+f'lvl_{lvl}_elastic_edges_{force}.pickle',
            base_path+f'lvl_{lvl}_elastic_no_cable_{force}.pickle',
            *[base_path+f'lvl_{lvl}_visco_phi0={phi0}_{force}.pickle' for phi0 in phi0s],
            *[base_path+f'lvl_{lvl}_visco_no_cable_phi0={phi0}_{force}.pickle' for phi0 in phi0s]
            ] for force in forces
        ]
    # funcs=[elastic_func, lambda f: run(f, cable=False)]
    # savepaths = [
    #     [base_path+f'elastic_edges_{force}.pickle',
    #     base_path+f'elastic_no_cable_{force}.pickle',
    #     ] for force in forces
    # ]

    if viewable:
        parameter_sweep_analyzer(forces, funcs, plot_results, 
                                pre_process=shortest_length,
                                savepaths=savepaths, 
                                overwrite=False, 
                                inpaint=np.nan,
                                cache=True)
    else:
        parameter_sweep(forces, funcs, savepaths=savepaths, overwrite=False)
    # run(forces[-2], visco=False, phi0=0.3, cable=False)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

