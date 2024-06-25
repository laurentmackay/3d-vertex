import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull



from VertexTissue.Sweep import sweep

from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask, first_dict_value
from VertexTissue.TissueForces import compute_network_indices

from Step2_bis import depth_timeline, final_arc_ratio, final_depth, final_inter_arc_distance, final_width, inter_arc_distance_timeline, intercalations, run, phi0s,  base_path, kws_baseline,  clinton_double, final_angle, L0_T1s, clinton_middle, clinton_outer, final_inter_arc_depth, final_depth2, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline

from Step2_bis import kws_middle, kws_double, kws_outer


from VertexTissue.Energy import get_cell_volumes, network_energy

fontsize=14

def pressure_energy_timeseries(d,phi0=1.0,**kw):
    # G0=first_dict_value(d)
    # _, _, _, _, triangle_inds, triangles_sorted, _ = compute_network_indices(G0) 
    # triangulation= (triangle_inds, triangles_sorted)

    
    U_vec = np.array([ (t,network_energy(G,phi0=phi0, ec=0.2, get_volumes=get_cell_volumes,  bending=False, spring=False)) for t,G in d.items()])

    return U_vec


if __name__ == '__main__':

        remodel=False
        L0_T1=L0_T1s
        idx=-1

    

        def plot_timeseries(xy, index=0, baseline=None, linestyle='-'):
                for j in range(len(phi0s)):
                        x=xy[j,index][:,0]
                        y=xy[j,index][:,1]
                        if baseline is None:
                            plt.plot(x,y, label=f'$\phi_0={phi0s[j]}$', linestyle=linestyle)
                        else:
                            if np.isscalar(baseline):
                                xb=xy[j,baseline][:,0]
                                yb=xy[j,baseline][:,1]
                            else:
                                xb=baseline[:,0]
                                yb=baseline[:,1]   
                            plt.plot(x,y-np.interp(x,xb,yb), label=f'$\phi_0={phi0s[j]}$')


        def plot0(y,y2=None):
                p=plt.plot(phi0s, y) 
                if y2 is not None:
                        plt.plot(phi0s, y2, linestyle='-.', color=p[0].get_color()) 

                plt.xlabel('$\phi_0$')
                plt.xlim(min(phi0s), max(phi0s))

        def plot(y):
                for j in range(len(phi0s)):
                        plt.plot(intercalations, y[j,:], label=f'$\phi_0={phi0s[j]}$')      
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))

        def plot2(y1,y2):
                for j in range(len(phi0s)):
                        p=plt.plot(intercalations, y1[j,:], label=f'$\phi_0={phi0s[j]}$')
                        plt.plot(intercalations, y2[j,:], color=p[0].get_color(), linestyle='-.')
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))

        
                
        refresh=False

        energy_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = pressure_energy_timeseries,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        energy_middle  = sweep(phi0s, run, kw=kws_middle, pre_process = pressure_energy_timeseries,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        

        energy_outer  = sweep(phi0s, run, kw=kws_outer, pre_process = pressure_energy_timeseries,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        

        energy_double  = sweep(phi0s, run, kw=kws_double, pre_process = pressure_energy_timeseries,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)







        refresh=False
        
        # depth_middle_remodel  = sweep(phi0s, run, kw=naught_middle_remodel, pre_process = final_depth2,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # depth_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = final_depth2, 
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # depth_double_remodel  = sweep(phi0s, run, kw=naught_double_remodel, pre_process = final_depth2,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # plt.plot(phi0s, depth_baseline, label='baseline')
        base = energy_baseline
        mid = energy_middle
        outer = energy_outer
        double = energy_double

        # mid_remodel = depth_middle_remodel
        # outer_remodel = depth_outer_remodel
        # double_remodel = depth_double_remodel




        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(12.5, 4)
        # # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()
        plt.figure()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')
        idx=4
        baseline=0
        # baseline=energy_baseline[0]
        # plt.sca(axs[0])
        plt.title('Middle Region')
        plot_timeseries(mid, index=idx, baseline=baseline)
        # plot_timeseries(mid, index=idx, baseline=baseline)
        # plt.ylabel('$\Delta\;depth\;(\mu $m)')

        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot_timeseries(outer, index=idx, baseline=baseline)
        # #plt.ylabel('$\Delta\;depth\;(\mu $m)')

        # plt.sca(axs[2])
        # plt.title('Both')
        # # plot2(double-depth_baseline[0],mid+outer-2*depth_baseline[0])
        # plot_timeseries(double, index=idx, baseline=baseline)
        # #plt.ylabel('$\Delta\;depth\;(\mu $m)')

        
        # for i in range(double.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(double[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
                            
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout()   
        # plt.savefig('invagination_depth_vs_intercalations.png',dpi=200)
        plt.show()

