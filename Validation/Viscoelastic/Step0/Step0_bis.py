import os

from matplotlib import colors
import VertexTissue
from VertexTissue.Memoization import function_call_savepath

from VertexTissue.visco_funcs import SLS_nonlin, crumple, fluid_element
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

from Validation.Viscoelastic.Step2.Step2_bis import ecs

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

def run(force,  phi0=1.0, level=0, arcs=3, visco=True, cable=True, verbose=False, ec=0.2, contract=True, extend=False,
        SLS=False,SLS_no_extend=False, SLS_no_contract=False, fastvol=False):

    if SLS is False:
        k_eff = (phi0-ec)/(1-ec)
    else:
        k_eff=phi0

    # if k_eff<=0.01:
    #        return
    
    if contract==False and extend==False:
        return
    
    # if phi0==1:
    #     return
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
        kw={'rest_length_func':fluid_element(phi0=phi0, ec=ec, contract=contract, extend=extend) if not SLS else None}

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
    # def SLS_nonlin(ell, L, L0):
    #     eps=np.zeros(L.shape)
    #     dLdt=np.zeros(L.shape)

    #     inds = L>0

    #     eps[inds]=(ell[inds]-L0[inds])/L0[inds]

    #     inds = np.logical_and( L<=0, ell>0)
    #     if SLS_no_extend:
    #             inds = np.logical_or(np.logical_or(inds ,  eps<-ec),  ell>L)
    #     else:
    #             inds = np.logical_or(np.logical_or(inds ,  eps>ec),  ell<L)

    #     dLdt[inds] = (ell[inds]-L[inds])

    #     return dLdt
    
    if (SLS and (SLS_no_contract or SLS_no_extend or ec!=0.0)):
        maxwell_nonlin = SLS_nonlin(ec=ec, contract=not SLS_no_contract, extend=not SLS_no_extend)
    else:
        maxwell_nonlin=None
        
    integrate = monolayer_integrator(G, G_apical, 
                                    pre_callback=squeeze, 
                                    SLS = False if not SLS else phi0,
                                    intercalation_callback=terminate, 
                                    termination_callback=wait_for_intercalation,  
                                    blacklist=True, length_abs_tol=1e-2,
                                    maxwell_nonlin=maxwell_nonlin, fastvol=fastvol,
                                    player=False, viewer={'button_callback':terminate} if viewable else False, minimal=False, **kw)
    #{'button_callback':terminate,'nodeLabels':None} if viewable else False
    #integrates
    # try:
    # print(belt)
    pattern=os.path.join(base_path, function_call_savepath()+'.pickle')

    # pattern=None
    print(f'starting f={force}')
    if not SLS:
        k_eff = (phi0-ec)/(1-ec)
    else:
        k_eff=phi0

    dt_min = 1e-3 if not cable else 1e-4

    integrate(5, 8000, 
            dt_init = 1e-4,
            adaptive=True,
            dt_min=max(dt_min*k_eff,0.2*dt_min),
            save_rate=-1,
            view_rate=20,
            verbose=verbose,
            save_pattern=pattern)
    
    print('done')
    # except:
    #     print(f'failed to integrate tau={tau}')
    
    #     pass

def shortest_length(d, e=None, level=None, **kw):
    if e is None:
        e=inter_edges[level]
    return min([ euclidean_distance(G.nodes[e[0]]['pos'],G.nodes[e[1]]['pos']) for G in d.values() ])

def process_results(forces, results, plot=False):

    lens = results
    
    lens_no_cable = lens[:,0,:,:]
    lens_cable = lens[:,1,:,:]

    contraction_force_cable = np.tile(np.nan,lens.shape[2:])
    contraction_force_no_cable = np.tile(np.nan,lens.shape[2:])

    for i in range(lens.shape[2]):
        for j in range(lens.shape[3]):
            contracted_no_cable=np.argwhere(lens_no_cable[:,i,j]< (1.01)* const.l_intercalation)
            
            if len(contracted_no_cable):
                contraction_force_no_cable[i,j] = forces[contracted_no_cable[0,0]]

            contracted_cable=np.argwhere(lens_cable[:,i,j]< (1.01)*const.l_intercalation)
            if len(contracted_cable):
                contraction_force_cable[i,j] = forces[contracted_cable[0,0]]

    
    if plot:
         
        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(9.5, 3.5)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        plt.sca(axs[0])
        plt.pcolor(plot[0], plot[1], contraction_force_no_cable)
        plt.colorbar()

        plt.sca(axs[1])
        plt.pcolor(plot[0], plot[1], contraction_force_cable)
        plt.colorbar()

        plt.show()

    return contraction_force_no_cable*const.myo_beta, contraction_force_cable*const.myo_beta

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

    # plt.savefig(f'triple_belt_lvl_{level}_cable_contraction_127.pdf',dpi=200)
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
    
    
    

phi0s=list(reversed([ 0.0, 0.025, 0.05, 0.1, 0.15, 0.2, .3,  .4, .5,  .6, .7, .8, .9, 1.0]))
# phi0s=list(reversed([  .3,  .4, .5,  .6, .7, .8, .9, 1.0]))
# colors=['k','r','g','b','y','m','c','orange']
def main():
    forces=np.array([ *np.linspace(0,850,40), *np.linspace(850,1200,20)[1:]])
    forces = np.linspace(0, 600, 80)
    
    # forces=np.linspace(0,850,60)



    kws={'cable':[False,True], 'level':[0,2,3],  'phi0':phi0s}
    kws_thresh={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'extend':[False,True], 'contract':[False,True]}
    kws_thresh_sym={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'extend':True, 'contract':True}
    kws_thresh_con={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'extend':False, 'contract':True}
    kws_thresh_ext={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'extend':True, 'contract':False}

    kws_SLS_thresh_sym={'cable':[False,True], 'level':1,  'phi0':np.flip(phi0s), 'ec':ecs, 'SLS':True}
    kws_SLS_thresh_con={'cable':[False,True], 'level':1,  'phi0':np.flip(phi0s), 'ec':ecs,'SLS':True, 'SLS_no_extend':True}
    kws_SLS_thresh_ext={'cable':[False,True], 'level':1,  'phi0':np.flip(phi0s), 'ec':ecs, 'SLS':True, 'SLS_no_contract':True}

    kws_SLS_thresh_sym={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'SLS':True, 'fastvol':[False,True]}
    kws_SLS_thresh_con={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs,'SLS':True, 'SLS_no_extend':True, 'fastvol':[False,True]}
    kws_SLS_thresh_ext={'cable':[False,True], 'level':1,  'phi0':phi0s, 'ec':ecs, 'SLS':True, 'SLS_no_contract':True, 'fastvol':[False,True]}


    kws_no_cable_0={'cable':[False], 'level':0,  'phi0':phi0s}
    kws_no_cable_1={'cable':[False], 'level':1, 'phi0':phi0s}
    kws_no_cable_2={'cable':[False], 'level':2,  'phi0':phi0s}
    kws_no_cable_3={'cable':[False], 'level':3,  'phi0':phi0s}

    kws_0={'cable':[True], 'level':0,  'phi0':phi0s}
    kws_1={'cable':[True], 'level':1, 'phi0':phi0s}
    kws_2={'cable':[True], 'level':2,  'phi0':phi0s}
    kws_3={'cable':[True], 'level':3,  'phi0':phi0s}


    kws=kws_3
    

    viewable=False
    if viewable:
        refresh=False
        overwrite=False

        results_sym = sweep(forces, run,  kw = kws_SLS_thresh_sym,
        pre_process=shortest_length,
        savepath_prefix=base_path,
        overwrite=overwrite,
        refresh=refresh,
        cache=True,
        pass_kw=True,
        inpaint= np.nan)
        
        results_ext = sweep(forces, run,  kw = kws_SLS_thresh_ext,
        pre_process=shortest_length,
        savepath_prefix=base_path,
        overwrite=overwrite,
        refresh=refresh,
        cache=True,
        pass_kw=True,
        inpaint= np.nan)

        results_con = sweep(forces, run,  kw = kws_SLS_thresh_con,
        pre_process=shortest_length,
        savepath_prefix=base_path,
        overwrite=overwrite,
        refresh=refresh,
        cache=True,
        pass_kw=True,
        inpaint= np.nan)

        force_no_cable_con, force_cable_con = process_results(forces, results_con)
        force_no_cable_ext, force_cable_ext = process_results(forces, results_ext)
        force_no_cable_sym, force_cable_sym = process_results(forces, results_sym)

        fig, axs = plt.subplots(2,3)
        fig.set_size_inches(10, 8)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()

        vmin = np.nanmin((np.nanmin(force_no_cable_con), np.nanmin(force_no_cable_ext), np.nanmin(force_no_cable_sym)))

        vmax = np.nanmax((np.nanmax(force_no_cable_con), np.nanmax(force_no_cable_ext), np.nanmax(force_no_cable_sym)))

        class nf(float):
            def __repr__(self):
                s = f'{self:.1f}'
                return f'{self:.0f}' if s[-1] == '0' else s
        
        def contour_padded(x,y,z, **kw):
            pad = ((1,1),(1,1))
            delta=0.0001
            xvals=((x[0]-(x[1]-x[0])*delta, x[-1] + (x[-1]-x[-2])*delta))
            yvals=((y[0]-(y[1]-y[0])*delta, y[-1] + (y[-1]-y[-2])*delta))
            plt.xlim((x[0],x[-1]))
            plt.ylim((y[-1],y[0]))
            CS = plt.contour(np.pad(x,(1,1), constant_values=xvals), np.pad(y,(1,1), constant_values=yvals), np.pad(z,pad,mode='edge'),**kw)
            plt.xlim((x[0],x[-1]))
            plt.ylim((y[-1],y[0]))

            CS.levels = [nf(val) for val in CS.levels]
            if plt.rcParams["text.usetex"]:
                fmt = r'%r nN'
            else:
                fmt = '%r nN'

            plt.gca().clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing =5)

            
        plt.sca(axs[0])
        plt.pcolormesh(ecs, phi0s, force_no_cable_con, vmin=vmin, vmax=vmax, shading='gouraud')
        contour_padded(ecs, phi0s, force_no_cable_con, colors='k', extent=(0,0.25,0.1,1), levels=[2.50, 3.00, 3.75 ,4.50, 5.50])
        # plt.colorbar()

        plt.sca(axs[1])
        # plt.pcolor(ecs, phi0s, force_no_cable_ext, vmin=vmin, vmax=vmax)
        plt.pcolormesh(ecs, phi0s, force_no_cable_ext, vmin=vmin, vmax=vmax, shading='gouraud')
        contour_padded(ecs, phi0s, force_no_cable_ext, colors='k', extent=(0,0.25,0.1,1), levels=[4.50, 4.75, 5.00, 5.25, 5.50])
        # plt.colorbar()

        plt.sca(axs[2])
        plt.pcolormesh(ecs, phi0s, force_no_cable_sym, vmin=vmin, vmax=vmax, shading='gouraud')
        #         # plt.pcolor(ecs, phi0s, force_no_cable_sym, vmin=vmin, vmax=vmax)
        

        box = axs[2].get_position()
        delta = 0.05
        axColor = plt.axes([box.x0 + box.width * (1+delta), box.y0, 0.01, box.height])
        plt.colorbar( cax = axColor, orientation="vertical")
        plt.sca(axs[2])
        contour_padded(ecs, phi0s, force_no_cable_sym, colors='k', extent=(0,0.25,0.1,1), levels=[1.5, 2.0, 3.00, 3.75 ,4.50, 5.50])

               
        vmin = np.nanmin([np.nanmin(x) for x in (force_cable_con, force_cable_ext, force_cable_sym)])

        vmax = np.nanmax([np.nanmax(x) for x in (force_cable_con, force_cable_ext, force_cable_sym)])

        plt.sca(axs[3])
        plt.pcolor(ecs, phi0s, force_cable_con, vmin=vmin, vmax=vmax)
        # plt.pcolormesh(ecs, phi0s, force_cable_con, vmin=vmin, vmax=vmax, shading='gouraud')
        contour_padded(ecs, phi0s, force_cable_con, colors='k', extent=(0,0.25,0.1,1), levels=[2.50, 3.00, 3.75 ,4.50])
        # plt.colorbar()

        plt.sca(axs[4])
        plt.pcolor(ecs, phi0s, force_cable_ext, vmin=vmin, vmax=vmax)
        # plt.pcolormesh(ecs, phi0s, force_cable_ext, vmin=vmin, vmax=vmax, shading='gouraud')
        contour_padded(ecs, phi0s, force_cable_ext, colors='k', extent=(0,0.25,0.1,1), levels=[2.50, 3.00, 3.75 ,4.50])
        # plt.colorbar()

        plt.sca(axs[5])
        plt.pcolor(ecs, phi0s, force_cable_sym, vmin=vmin, vmax=vmax)

        box = axs[5].get_position()
        delta = 0.05
        axColor = plt.axes([box.x0 + box.width * (1+delta), box.y0, 0.01, box.height])
        plt.colorbar( cax = axColor, orientation="vertical")
        
        plt.show()

    else:
        results = sweep(np.flip(forces), run,
        kw = kws_SLS_thresh_ext,
        pre_process=shortest_length,
        savepath_prefix=base_path,
        overwrite=False,
        cache=True,
        pass_kw=True,
        inpaint= np.nan if viewable else None)



    print('done')

if __name__ == '__main__':
    main_pid = os.getpid()
    main()

