from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


from Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder, final_buckle_angle
from VertexTissue.Sweep import sweep
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask
from VertexTissue.Memoization import get_caller_locals
from VertexTissue.Tissue import tissue_3d, get_outer_belt
from VertexTissue.util import pcolor

from Step2_bis import  inter_arc_distance, run, phi0s,  base_path, kws, kws_baseline, kws_middle, kws_outer, kws_double, get_inter_edges, L0_T1s


fontsize=14
from Step2_bis import depth_timeline, intercalations, run, phi0s,  base_path, kws_baseline,  clinton_double, final_angle, L0_T1s, clinton_middle, clinton_outer, final_inter_arc_depth, final_depth2, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline
from Step2_bis import kws_baseline, kws_outer, kws_middle, kws_double

if __name__ == '__main__':

        def middle_depth(d):
                return inter_arc_distance(last_dict_value(d), outer=False)
        
        def outer_depth(d):
                return inter_arc_distance(last_dict_value(d), outer=True)

        refresh=False
        depth_func  = final_depth2

        depth_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        depth_middle  = sweep(phi0s, run, kw=kws_middle, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        depth_outer  = sweep(phi0s, run, kw=kws_outer, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        depth_double  = sweep(phi0s, run, kw=kws_double, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        mid = depth_middle
        outer = depth_outer
        double = depth_double



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

        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(15, 5)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')

        plt.sca(axs[0])
        plt.title('Middle Region')
        plot(L0_T1s,mid-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')

        
        plt.sca(axs[1])
        plt.title('Outer Region')
        plot(L0_T1s,outer-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')

        
        plt.sca(axs[2])
        plt.title('Both')
        plot(L0_T1s,double-depth_baseline[0], xlabel = '$L_0^{T1}$', ylabel = '$\Delta\;depth\;(\mu $m)')


        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.savefig('invagination_depth_vs_rest_length.pdf',dpi=200)
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