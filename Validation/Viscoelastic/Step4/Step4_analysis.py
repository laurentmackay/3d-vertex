from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


from Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder, final_buckle_angle
from Validation.Viscoelastic.Step2.Step2_bis import angle_timeseries, extension_timeseries
from VertexTissue.Sweep import sweep
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask
from VertexTissue.Memoization import get_caller_locals
from VertexTissue.Tissue import tissue_3d, get_outer_belt
from VertexTissue.util import pcolor

from Validation.Viscoelastic.Step4.Step4 import  inter_arc_distance, run, phi0s,  base_path, kws, kws_baseline, kws_middle, kws_outer, kws_double, get_inter_edges, L0_T1s


fontsize=14
from Validation.Viscoelastic.Step4.Step4 import depth_timeline, intercalations, run, phi0s,  base_path, kws_baseline,  clinton_double, final_angle, L0_T1s, clinton_middle, clinton_outer, final_inter_arc_depth, final_depth, final_lumen_depth
from Validation.Viscoelastic.Step4.Step4 import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline
from Validation.Viscoelastic.Step4.Step4 import kws_baseline, kws_outer, kws_middle, kws_double

if __name__ == '__main__':

        def middle_depth(d):
                return inter_arc_distance(last_dict_value(d), outer=False)
        
        def outer_depth(d):
                return inter_arc_distance(last_dict_value(d), outer=True)

        refresh=False
        N=12
        # kws_middle_slice = kws_middle.copy()
        # kws_middle_slice['intercalations']=N

        # kws_outer_slice = kws_outer.copy()
        # kws_outer_slice['intercalations']=N

        # angle_middle  = sweep(phi0s, run, kw=kws_middle_slice, pre_process = angle_timeseries, pass_kw=True, pre_process_kw={'basal': False, 'summary':np.mean, 'tmin':800},
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        # angle_outer  = sweep(phi0s, run, kw=kws_outer_slice, pre_process = angle_timeseries, pass_kw=True, pre_process_kw={'basal': False, 'summary':np.mean, 'tmin':800},
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        # def plot_timeseries(xy):
        #         for j in range(len(phi0s)):
        #                 if xy[j] is not None:
        #                         p=plt.plot(xy[j][:,0], xy[j][:,1], label=f'$\phi_0={phi0s[j]}$')

        # plot_timeseries(angle_middle)
        # plt.show()

        # plot_timeseries(angle_outer)
        # plt.show()



        refresh=False
        depth_func  = final_depth

        depth_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        depth_middle  = sweep(phi0s, run, kw=kws_middle, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        depth_outer  = sweep(phi0s, run, kw=kws_outer, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        # depth_double  = sweep(phi0s, run, kw=kws_double, pre_process = depth_func ,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        base = depth_baseline.reshape((-1,1))
        mid = depth_middle
        outer = depth_outer
        # double = depth_double



        def plot(x,y, flatten=False, xlabel=None, ylabel=None):
                for j in range(len(phi0s)):
                        if flatten:
                                plt.errorbar(x, np.nanmean(y[j,:,:],axis=(0,1)), yerr=np.nanstd(y[j,:,:],axis=(0,1)), label=f'$\phi_0={phi0s[j]}$', capsize=4)
    
                        else:
                                plt.errorbar(x, np.nanmean(y[j,:,:],axis=0), yerr=np.nanstd(y[j,:,:],axis=0), label=f'$\phi_0={phi0s[j]}$', capsize=4)
                if isinstance(x, Iterable):
                        plt.xlim(min(x), max(x)*1.02)
                if xlabel is not None:
                        plt.xlabel(xlabel)
                if ylabel is not None:
                        plt.ylabel(ylabel)
        # plt.figure()
        # plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')


        fig, axs = plt.subplots(1,2)
        fig.set_size_inches(9.5, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()

        N=len(phi0s) 
        colors = plt.cm.nipy_spectral(np.linspace(0,.85,N))
                # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors )
                
        plt.sca(axs[0])
        plt.title('Middle Region')
        for j in range(len(phi0s)):
                plt.plot(intercalations, mid[j,:]-base[j], label=f'$\phi_0={phi0s[j]}$', color=colors[j])
        # plt.legend()
        # plt.show()
        plt.xlim(min(intercalations), max(intercalations))
        plt.ylim((0,2.85))
        plt.xlabel('# of intercalations')
        plt.ylabel('$\Delta\;depth\;(\mu $m)')
        plt.xticks(ticks=[0,5,10,15])
        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot(L0_T1s,outer-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')

        
        # plt.sca(axs[2])
        # plt.title('Both')
        # plot(L0_T1s,double-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')


        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        # plt.tight_layout() 
        # plt.savefig('invagination_depth_vs_rest_length.pdf',dpi=200)
        # plt.show()

        # plt.figure()
        # plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')
        plt.sca(axs[1])
        plt.title('Outer Region')
        for j in range(len(phi0s)):
                plt.plot(intercalations, outer[j,:]-base[j], label=f'$\phi_0={phi0s[j]}$', color=colors[j])
        # plt.legend()
        # plt.show()
        plt.xlim(min(intercalations), max(intercalations))
        plt.xlabel('# of intercalations')
        plt.ylim((0,2.85))
        plt.xticks(ticks=[0,5,10,15])
        
        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot(L0_T1s,outer-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')

        
        # plt.sca(axs[2])
        # plt.title('Both')
        # plot(L0_T1s,double-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')


        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout() 
        plt.savefig('invagination_depth_vs_rest_length_continuous_pressure.pdf',dpi=200)
        plt.show()



        # mid_depth_middle  = sweep(phi0s, run, kw={**kws_middle,**{'L0_T1':L0_T1s[-2:]}}, pre_process = middle_depth,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # out_depth_outer  = sweep(phi0s, run, kw={**kws_outer,**{'L0_T1':L0_T1s[-2:]}}, pre_process = outer_depth,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # mid_depth_double  = sweep(phi0s, run, kw={**kws_double,**{'L0_T1':L0_T1s[-2:]}}, pre_process = middle_depth,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # out_depth_double  = sweep(phi0s, run, kw={**kws_double,**{'L0_T1':L0_T1s[-2:]}}, pre_process = outer_depth,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # [  plot(x , y, flatten=True) for x,y in zip(  [1,2,3,4]      ,  (mid_depth_middle , out_depth_outer, mid_depth_double, out_depth_double ))]

        # plt.show()