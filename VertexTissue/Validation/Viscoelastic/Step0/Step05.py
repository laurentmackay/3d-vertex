import os

from matplotlib import colors
from Validation.Viscoelastic.Step2.Step2_bis import get_inter_edges
import VertexTissue
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.funcs_orig import clinton_timestepper, convex_hull_volume_bis, get_points

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

from VertexTissue.globals import default_ab_linker, default_edge, belt_strength, outer_arc, inner_arc, press_alpha
import VertexTissue.globals as const

from VertexTissue.util import edge_index, find_first, first, get_myosin_free_cell_edges,  inside_arc, shortest_edge_length_and_time, shortest_edge_network_and_time

from VertexTissue.Iterable import first_item, imin

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

from VertexTissue.Dict import last_dict_key, last_dict_value
from VertexTissue.Geometry import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)

# inter_edges = ((305, 248), (94,163), (69,8), (2,8))

def run(force,  phi0=1.0, level=0, arcs=2, visco=True, cable=True, verbose=False, 
        intercalations=1, outer=False, double=False, clinton_timestepping=False
        , press_alpha=press_alpha, pit_strength=0, continuous_pressure=False):

    #
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    belt = get_outer_belt(G_apical)


    #initialize some things for the callback

    const.press_alpha=press_alpha

    if arcs==0:
        arc_list=(belt,)
    if arcs==1:
        arc_list=(outer_arc,)
    elif arcs==2:
        arc_list=(outer_arc, inner_arc)
    elif arcs==3:
        arc_list=(outer_arc, inner_arc, belt)

    inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)
    
    # if clinton_timestepping:
    clinton_timestep, uncontracted = clinton_timestepper(G, inter_edges)
        
    squeeze = SG.arc_pit_and_intercalation(G, arc_list, t_belt=0 if pit_strength==0 else 375, inter_edges=inter_edges,  t_intercalate=0 if pit_strength==0 else 375, intercalation_strength=force,
                                        arc_strength = belt_strength if cable else 0.0, pit_strength=pit_strength)


    # const.press_alpha/=4
    if not visco:
        kw={}
    else:
        kw={'rest_length_func':crumple(phi0=phi0)}

    done=False

    def terminate(*args, **kw):
        nonlocal done
        done=True

    def wait_for_intercalation(*args):
        nonlocal done
        return done

    const.l_intercalation=0.1
    const.l_mvmt=0.001
    #create integrator
    N_cells=len(G.graph['centers'])
    v0=np.ones((N_cells,))*const.v_0

    def label_contracted(i,j,locals=None, **kw):
        nonlocal done
        inds = edge_index(G, inter_edges)
        k = find_first( edge_index(G, (i,j)) == inds )
        uncontracted[ k ] = False

        if np.all(np.array(uncontracted)==False):
            done=True

        if continuous_pressure:
            centers = locals['centers']
            PI = locals['PI']
            pos = locals['pos']
            four_cells = list({ k for k in G.neighbors(i) if k in centers}.union( { k for k in G.neighbors(j) if k in centers}))
            inds = [np.argwhere(centers == c)[0,0] for c in four_cells]
            curr_vols = [convex_hull_volume_bis(get_points(G, c, pos) ) for c in four_cells]
            v0[inds]=curr_vols+PI[inds]/const.press_alpha

    
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=label_contracted,
                                    termination_callback=wait_for_intercalation,  
                                    blacklist=True, length_abs_tol=1e-2,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None} if viewable else False,
                                    minimal=False, constant_pressure_intercalations=continuous_pressure, v0=v0 if continuous_pressure else const.v_0, **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')

    # pattern=None
    print(f'starting f={force}')
    ec=0.2
    k_eff = (phi0-ec)/(1-ec)

    dt_min = 5e-2

    integrate(5, 2000, 
            dt_init = 0.5 if clinton_timestepping else 1e-3,
            adaptive=True,
            dt_min=dt_min*k_eff,
            timestep_func=clinton_timestep if clinton_timestepping else None,
            adaptation_rate=1 if clinton_timestepping else 0.1,
            save_rate=100,
            verbose=verbose,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def completed_intercalations(d, intercalations=1, outer=False, double=False, tmax=600, max_spread=np.inf, **kw):
    inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)
    G=last_dict_value(d)
    tfinal = last_dict_key(d)
    if tfinal>tmax:
        return False
        
    for e in inter_edges:
        i=e[0]
        j=e[1]
        if  G[i][j]['myosin']>0.01:
            return False

    if np.isfinite(max_spread) and intercalations>1:
        iter = reversed(d.keys())
        t_prev=next(iter)
        G_prev=G
        uncontracted_prev=0
        for t in iter:
            G=d[t]
            uncontracted=sum([G[e[0]][e[1]]['myosin']>0.01  for e in inter_edges])

            if uncontracted==intercalations:
                if uncontracted_prev>0:
                    return (tfinal-t_prev)<=max_spread
                else:
                    return True
            uncontracted_prev=uncontracted
            t_prev=t


    # print('oh lordy he coming', last_dict_key(d))
    return True

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
    
    
def find_min_force(mask, forces, inpaint=np.nan):
    contraction_force = np.zeros(mask.shape[1:] if len(mask.shape)>1 else 1)
    contraction_force[:]=inpaint
    forces_raveled = contraction_force.T
    mask_reshaped = np.swapaxes(mask, 0,-1)
    for i, m in enumerate(mask_reshaped):
        if np.any(m==1.0) :
            forces_raveled.flat[i]=forces[find_first(m==1.0)]
        elif np.any(m==True):
            forces_raveled.flat[i]=forces[find_first(m==True)]
        else:
            forces_raveled.flat[i]=np.nan
        
    return contraction_force




phi0s=list(reversed([ .22, .25, .3,  .4, .5,  .6, .7, .8, .9, .95, 1.0]))
# colors=['k','r','g','b','y','m','c','orange']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(400, 1000, 80)
    
    # forces=np.linspace(0,850,60)

   
    intercalations = [1, 4, 6, 8, 12, 14, 16,18]
    kws={'cable':[True], 'intercalations':intercalations, 'press_alpha':[0.046, 0.00735],  'phi0':[1.0,], 'clinton_timestepping':[True, False], 'pit_strength':540, 'continuous_pressure':[True, False]}

    kws_new={ 'cable':[True], 'intercalations':intercalations, 'press_alpha':[0.00735],  'phi0':[1.0,], 'clinton_timestepping':False, 'pit_strength':540, 'continuous_pressure':False}
    kws_orig={'cable':[True], 'intercalations':intercalations, 'press_alpha':[0.046],    'phi0':[1.0,],  'clinton_timestepping':True, 'pit_strength':540, 'continuous_pressure':False}


    # run(800, intercalations=12, press_alpha=0.00735, clinton_timestepping=False, verbose=True, pit_strength=540, continuous_pressure=True)

    spread=12
    
    mask_orig = np.squeeze(sweep(forces, run,  kw = kws_orig,
        pre_process=completed_intercalations,
        savepath_prefix=base_path,
        overwrite=False,
        cache=True,
        refresh=False,
        pass_kw=True,
        pre_process_kw={'tmax':2000, 'max_spread':spread},
        inpaint= np.nan if viewable else None))

    mask_new = np.squeeze(sweep(forces, run,  kw = kws_new,
        pre_process=completed_intercalations,
        savepath_prefix=base_path,
        overwrite=False,
        cache=True,
        refresh=False,
        pass_kw=True,
        pre_process_kw={'tmax':2000, 'max_spread':spread},
        inpaint= np.nan if viewable else None))


    threshold_new = find_min_force(mask_new  , forces)
    threshold_orig = find_min_force(mask_orig, forces)
    if np.isnan(threshold_orig[-1]):
        threshold_orig[-1]=1020

    plt.plot(intercalations, threshold_orig, label='Required for rapid simultaneous contraction', linewidth=3)

    # plt.plot(intercalations, threshold_new)

    # plt.plot(np.array(intercalations)[[0,-1]],[560,560], linestyle='-',color='k')

    plt.plot([0,intercalations[-1]],[750,750], linestyle='--',color='k',label="Clinton's baseline")

    # plt.plot(np.array(intercalations)[[0,-1]],[1080,1080], linestyle='-',color='k')

    ylim=plt.ylim()
    plt.ylim((0,ylim[1]))
    plt.xlim((0,18))
    plt.xticks(intercalations)
    plt.yticks([0,200,400,600,800,1000])
    plt.ylabel('myosins per edge', fontsize=16)
    plt.xlabel('# of T1 intercalations', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'clinton_contraction_myosin.pdf')
    plt.show()

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

