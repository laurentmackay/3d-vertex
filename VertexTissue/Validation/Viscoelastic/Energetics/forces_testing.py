import os

from matplotlib import colors
from Validation.Viscoelastic.Energetics.energy_testing_step_2 import cell_inds_touching_edge
from Validation.Viscoelastic.Step2.Step2_bis import get_inter_edges

import VertexTissue
from VertexTissue.Iterable import first_item
from VertexTissue.Memoization import function_call_savepath
from VertexTissue.TissueForces import compute_network_indices, pressure_forces, spring_forces
from VertexTissue.funcs_orig import convex_hull_volume_bis, get_points
from VertexTissue.visco_funcs import crumple

print('this is some basic output')



import numpy as np


import VertexTissue.globals as const
import VertexTissue.SG as SG


from VertexTissue.globals import belt_strength, inner_arc, outer_arc

from VertexTissue.Sweep import sweep
from VertexTissue.Tissue import  get_outer_belt, tissue_3d
from VertexTissue.util import first
from VertexTissue.vertex_3d import monolayer_integrator

from VertexTissue.Energy import get_cell_volumes, network_energy

try:
    import matplotlib.pyplot as plt

    from VertexTissue.PyQtViz import edge_view
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'

from VertexTissue.Dict import first_dict_value, last_dict_value
from VertexTissue.Geometry import euclidean_distance


def is_subprocess():
    return main_pid != os.getpid()

def decrease_nice():
    pid = os.getpid()
    os.system("sudo renice -n -19 -p " + str(pid))

dt=0.1
taus = np.logspace(6,1,5)
lvl=3
# inter_edges = ((305, 248), (94,163), (69,8), (2,8))

def run(force, visco=False,  phi0=1.0, level=0, arcs=2, cable=True, continuous_pressure=False,
        intercalations=1, t_final=600, pit_strength=540):

    inter_edges = get_inter_edges(intercalations=intercalations, outer=False, double=False)
    G, G_apical = tissue_3d( hex=7,  basal=True)
    
    belt = get_outer_belt(G_apical)


    #initialize some things for the callback

    
    if arcs==0:
        arc_list=(belt,)
    if arcs==1:
        arc_list=(outer_arc,)
    elif arcs==2:
        arc_list=(outer_arc, inner_arc)
    elif arcs==3:
        arc_list=(outer_arc, inner_arc, belt)

    squeeze = SG.arcs_pit_and_intercalation(G, arc_list, t_belt=3500, inter_edges=inter_edges,  t_intercalate=375, t_1=375, intercalation_strength=force,
                                        arc_strength = belt_strength if cable else 0.0, pit_strength=pit_strength)


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

    N_cells=len(G.graph['centers'])
    v0=np.ones((N_cells,))*const.v_0

    def constant_pressure(i,j,locals=None, **kw):

        centers = locals['centers']
        PI = locals['PI']
        pos = locals['pos']
        four_cells = list({ k for k in G.neighbors(i) if k in centers}.union( { k for k in G.neighbors(j) if k in centers}))
        inds = [np.argwhere(centers == c)[0,0] for c in four_cells]
        curr_vols = [convex_hull_volume_bis(get_points(G, c, pos) ) for c in four_cells]
        v0[inds]=curr_vols+PI[inds]/const.press_alpha


    const.l_intercalation=0.1
    const.l_mvmt=0.1
    #create integrator
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    intercalation_callback=constant_pressure if continuous_pressure else None, 
                                    termination_callback=wait_for_intercalation,  
                                    blacklist=True,
                                    player=False, viewer={'button_callback':terminate, 'nodeLabels':None} if viewable else False,
                                    minimal=False,
                                    constant_pressure_intercalations=continuous_pressure,
                                    v0=v0 if continuous_pressure else const.v_0,
                                    **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')

    # pattern=None
    print(f'starting f={force}')
    integrate(1, t_final, 
            dt_init = 1e-3,
            adaptive=True,
            dt_min=5e-2,
            save_rate=t_final/1000,
            verbose=True,
            save_pattern=pattern)
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass



def plot_results(forces,results):
    if not viewable:
        return
    
    lens = results/const.l_apical
    # times = np.reshape([e[1] for e in results.ravel()], results.shape)
    inter_len = const.l_intercalation/const.l_apical

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



def energy_timeseries(d,phi0=1.0,**kw):
    G0=first_dict_value(d)
    _, _, _, _, triangle_inds, triangles_sorted, _ = compute_network_indices(G0) 
    triangulation= (triangle_inds, triangles_sorted)

    
    U_vec = np.array([ network_energy(G,phi0=phi0, ec=0.2, triangulation=triangulation,  bending=False) for G in d.values()])

    return U_vec

def spring_energy_timeseries(d,phi0=1.0,**kw):
    G0=first_dict_value(d)

    
    U_vec = np.array([ (t,network_energy(G,phi0=phi0, ec=0.2,  get_volumes=get_cell_volumes,  bending=False, pressure=False)) for t,G in d.items()])


    return U_vec

def pressure_energy_timeseries(d, summary=np.max, tmin=0, tmax=np.inf, nodes=None, phi0=1.0,**kw):
    for t,G0 in d.items():
        if t>=tmin:
            print(t,t>tmin)
            break
    # G0=first_dict_value(d)
    ab_face_inds, side_face_inds, _, _, _, _, _ = compute_network_indices(G0) 
    faces=(ab_face_inds, side_face_inds)
    # p0 = pressure_forces(G0,faces=faces)
    v0=const.v_0
    series=[]
    for t,G in d.items():
        if t>tmax:
            break
        if 'v0' in G.graph.keys():
            # print([v0,])
            v0=G.graph['v0']
            ab_face_inds, side_face_inds, _, _, _, _, _ = compute_network_indices(G) 
            # faces=(ab_face_inds, side_face_inds)

        value = network_energy(G,phi0=phi0, ec=0.2, get_volumes=get_cell_volumes,  bending=False, spring=False, v0=v0)
        
        series.append((t,value))



    return np.array(series)



# def pressure_energy_timeseries(d,phi0=1.0,**kw):
#     G0=first_dict_value(d)

    
#     U_vec = np.array([ network_energy(G,phi0=phi0, ec=0.2, get_volumes=get_cell_volumes,  bending=False, spring=False) for G in d.values()])


#     return U_vec

def spring_force_timeseries(d,phi0=1.0, summary=np.max, tmax=np.inf ,nodes=None ,**kw):


    rest_length_func=crumple(phi0=phi0)
    series=[]
    for t,G in d.items():
        if t>tmax:
            break
        forces=np.sqrt(np.sum(spring_forces(G,rest_length_func=rest_length_func)**2,axis=1))

        if nodes is None:
            value = summary(forces)
        else:
            value = summary(forces[nodes])

        series.append((t,value))


    return np.array(series)

def pressure_force_timeseries(d, summary=np.max, tmin=0, tmax=np.inf, nodes=None,**kw):
    for t,G0 in d.items():
        if t>=tmin:
            print(t,t>tmin)
            break
    # G0=first_dict_value(d)
    ab_face_inds, side_face_inds, _, _, _, _, _ = compute_network_indices(G0) 
    faces=(ab_face_inds, side_face_inds)
    # p0 = pressure_forces(G0,faces=faces)
    v0=const.v_0
    series=[]
    for t,G in d.items():
        if t>tmax:
            break
        if 'v0' in G.graph.keys():
            # print([v0,])
            v0=G.graph['v0']
            ab_face_inds, side_face_inds, _, _, _, _, _ = compute_network_indices(G) 
            faces=(ab_face_inds, side_face_inds)

        dp=np.sqrt(np.sum(pressure_forces(G, faces=faces, v0=v0)**2,axis=1))
        
        if nodes is None:
            print(nodes)
            value = summary(dp)
        else:
            # print(nodes)
            value = summary(dp[nodes])
        
        series.append((t,value))



    return np.array(series)



phi0s=list(reversed([ .22, .25, .3,  .4, .5,  .6, .7, .8, .9]))
colors=['k','r','g','b','y','m','c','orange']

def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(0, 800, 60)
    forces=[1000,]
    # forces=np.linspace(0,850,60)




    
    kws={'cable':[True,], 'level':1, 'visco':[True,], 'phi0':[1.0], 'continuous_pressure':True, 'intercalations':1, 't_final':3000, 'pit_strength':300}
    kws0={'cable':[True,], 'level':1, 'visco':[True,], 'phi0':[1.0], 'continuous_pressure':True, 'intercalations':0, 't_final':3000 , 'pit_strength':300}



    G, G_apical = tissue_3d( hex=7,  basal=True)

    from itertools import product

    inter_edges = get_inter_edges(intercalations=1, outer=False, double=False)

    
    nodes, _ =  cell_inds_touching_edge(G, *inter_edges[-1], G.graph['centers'])

    nodes=[int(n) for n in nodes]


    nodes=[int(n) for n in np.unique(np.array(G.graph['circum_sorted'])[_])]
    p0=sweep(forces, run,  kw = kws0, pre_process=pressure_energy_timeseries, pass_kw=True, 
        pre_process_kw={'summary':np.max, 'nodes':nodes}, savepath_prefix=base_path, 
        overwrite=False, cache=True, refresh=False)

    r0=sweep(forces, run,  kw = kws0, pre_process=spring_energy_timeseries, pass_kw=True, 
        pre_process_kw={'summary':np.max, 'nodes':nodes}, savepath_prefix=base_path, 
        overwrite=False, cache=True, refresh=False)

    p0=p0[0][0][0][0]
    r0=r0[0][0][0][0]


    plt.plot(p0[:,0], p0[:,1],linestyle='-',color='y',label='pressure force')
    plt.plot(r0[:,0], r0[:,1],linestyle='-',color='m',label='spring force')
    plt.legend(loc='lower right', fontsize=12)
    plt.ylabel(r'max force $\left(\mu \rm{N} \right)$', fontsize=14)
    plt.xlabel('time (s)', fontsize=14)

    plt.savefig(f'energy_tetrad_no_intercalations.pdf')
    plt.show()

    for i,c in tuple(product((4,),(True,False))):
        kws['intercalations']=i
        kws['continuous_pressure']=c

        inter_edges = get_inter_edges(intercalations=i, outer=False, double=False)

        
        nodes, _ =  cell_inds_touching_edge(G, *inter_edges[-1], G.graph['centers'])

        nodes=[int(n) for n in nodes]


        nodes=[int(n) for n in np.unique(np.array(G.graph['circum_sorted'])[_])]

        p0=sweep(forces, run,  kw = kws0, pre_process=pressure_energy_timeseries, pass_kw=True, 
            pre_process_kw={'summary':np.max,  'nodes':nodes}, savepath_prefix=base_path, 
            overwrite=False, cache=True, refresh=False)

        r0=sweep(forces, run,  kw = kws0, pre_process=spring_energy_timeseries, pass_kw=True, 
            pre_process_kw={'summary':np.max, 'nodes':nodes}, savepath_prefix=base_path, 
            overwrite=False, cache=True, refresh=False)

        p0=p0[0][0][0][0]
        r0=r0[0][0][0][0]

        p=sweep(forces, run,  kw = kws, pre_process=pressure_energy_timeseries, pass_kw=True, 
                pre_process_kw={'summary':np.max, 'nodes':nodes, 'tmin':0}, savepath_prefix=base_path, 
                overwrite=False, cache=True, refresh=False)


                
        r=sweep(forces, run,  kw = kws, pre_process=spring_energy_timeseries, pass_kw=True, 
            pre_process_kw={'summary':np.max,  'nodes':nodes}, savepath_prefix=base_path, 
            overwrite=False, cache=True, refresh=False)


        p=p[0][0][0][0]
        r=r[0][0][0][0]
        pmax = p[:,1].max()
        jump = pmax>10
        f, axs = plt.subplots(2 if jump else 1, 1, sharex=True,  gridspec_kw={'height_ratios': [1, 3] if jump else [1,]})

        if jump:
            axs=axs.flatten()
        else:
            axs=[axs]

        axs[-1].plot(p0[:,0], p0[:,1],linestyle='--',color='y')
        axs[-1].plot(r0[:,0], r0[:,1],linestyle='--',color='m')

        if jump:
            axs[0].set_ylim(.8*pmax, 1.1*pmax,) 
            axs[-1].set_ylim(0,1.05*max(r0[:,1].max(),r[:,1].max()))

        for ax in axs:
            ax.plot(p[:,0], p[:,1],linestyle='-',color='y',label='pressure force')
            ax.plot(r[:,0], r[:,1],linestyle='-',color='m',label='spring force')

        if jump:
            axs[0].spines['bottom'].set_visible(False)
            axs[-1].spines['top'].set_visible(False)
            axs[0].xaxis.tick_top()
            axs[0].tick_params(labeltop=False)  # don't put tick labels at the top
            axs[-1].xaxis.tick_bottom()

            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
            axs[0].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=axs[-1].transAxes)  # switch to the bottom axes
            axs[-1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            axs[-1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        
        axs[0].legend(loc='upper right', fontsize=12)
        plt.ylabel(r'max force $\left(\mu \rm{N} \right)$', fontsize=14)
        plt.xlabel('time (s)', fontsize=14)

        plt.savefig(f'energy_tetrad_intercalations_{i}_continuous_pressure_{c}.pdf')
        plt.show()

    # run(forces[-2], visco=True, phi0=0.95)
    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

