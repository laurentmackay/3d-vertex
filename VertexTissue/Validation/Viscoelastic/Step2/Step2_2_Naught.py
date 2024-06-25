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
from VertexTissue.globals import outer_arc, inner_arc
from Step2_bis import intercalations, run, phi0s,  base_path, kws, kws_baseline,  naught_double, get_inter_edges, L0_T1s, naught_middle, naught_outer, final_angle, final_inter_arc_depth, final_depth, naught_middle_remodel, naught_outer_remodel, naught_double_remodel


fontsize=14


if __name__ == '__main__':

        remodel=False
        L0_T1=L0_T1s
        idx=-1


        
        refresh=False


        angle_middle  = (180/np.pi)*sweep(phi0s, run, kw=naught_middle, pre_process = final_angle, pass_kw=True,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        angle_outer  = (180/np.pi)*sweep(phi0s, run, kw=naught_outer, pre_process = final_angle, pass_kw=True,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        angle_double  = (180/np.pi)*sweep(phi0s, run, kw=naught_double, pre_process = final_angle, pass_kw=True,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        
        refresh=False

        for j in range(len(phi0s)):
                plt.plot(intercalations, angle_middle[j,:], label=f'phi_0={phi0s[j]}')
        plt.legend()
        plt.show()

        for j in range(len(phi0s)):
                plt.plot(intercalations, angle_outer[j,:], label=f'phi_0={phi0s[j]}')
        plt.legend() 

        
        plt.show()
        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(12, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()
        # # for i in range(1):
        # #         plt.sca(axs[i])
        # #         pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(angle_middle))
        # #         # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.max(mid[:,:,i])-mid_remodel[:,:,i]))
        # #         plt.colorbar()
        # #         # plt.show()

        # # plt.show()

        # # fig, axs = plt.subplots(1,3)
        # # fig.set_size_inches(12, 4)
        # # axs=axs.ravel()
        # # plt.get_current_fig_manager().canvas.set_window_title('Outer')
        # # for i in range(1):
        # #         plt.sca(axs[i])
        # #         pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(angle_outer))
        # #         # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(outer[:,:,i])-outer_remodel[:,:,i]))
        # #         plt.colorbar()
        # #         # plt.show()

        # # plt.show()

        # # fig, axs = plt.subplots(1,3)
        # # fig.set_size_inches(12, 4)
        # # axs=axs.ravel()
        # # plt.get_current_fig_manager().canvas.set_window_title('Double')
        # # for i in range(1):
        # #         plt.sca(axs[i])
        # #         pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(angle_double))
        # #         # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
        # #         plt.colorbar()
        # #         # plt.show()

        # # plt.show()

        # extension_middle  = sweep(phi0s, run, kw=naught_middle, pre_process = extension_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh).reshape(len(phi0s),len(intercalations),order='F')

        # extension_outer  = sweep(phi0s, run, kw=naught_outer, pre_process = extension_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh).reshape(len(phi0s),len(intercalations),order='F')

        # extension_double  = sweep(phi0s, run, kw=naught_double, pre_process = extension_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh).reshape(len(phi0s),len(intercalations),order='F')

        colors=['r','g','b']

        tissue_extension_middle  = sweep(phi0s, run, kw=naught_middle, pre_process = final_inter_arc_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        tissue_extension_outer  = sweep(phi0s, run, kw=naught_outer, pre_process = final_inter_arc_depth, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        tissue_extension_double  = sweep(phi0s, run, kw=naught_double, pre_process = final_inter_arc_depth, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        fig, axs = plt.subplots(1,3)
        axs=axs.ravel()
        fig.set_size_inches(12, 4)
        plt.get_current_fig_manager().canvas.set_window_title('Tissue Extension')
        plt.sca(axs[0])
        for j in range(len(phi0s)):
                plt.plot(intercalations, tissue_extension_middle[j,:], label=f'phi_0={phi0s[j]}')
        # plt.legend()
        # plt.show()
        plt.xlim(min(intercalations), max(intercalations))
        plt.xlabel('# of intercalations')
        # plt.get_current_fig_manager().canvas.set_window_title('Outer (Tissue Extension)')
        plt.sca(axs[1])
        for j in range(len(phi0s)):
                plt.plot(intercalations, tissue_extension_outer[j,:], label=f'phi_0={phi0s[j]}')
        # plt.legend() 
        # plt.show()
        plt.xlim(min(intercalations), max(intercalations))
        plt.xlabel('# of intercalations')
        # plt.get_current_fig_manager().canvas.set_window_title('Double (Tissue Extension)')
        plt.sca(axs[2])
        for j in range(len(phi0s)):
                plt.plot(intercalations, tissue_extension_double[j,:], label=f'phi_0={phi0s[j]}')
        plt.xlim(min(intercalations), max(intercalations))
        plt.xlabel('# of intercalations')
        plt.legend() 
        plt.show()
        # for i in range(3):
        #         for entry in extension_middle[:2,i,:].T:
        #                 for _ in entry:
        #                         if _ is not None:
        #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # plt.show()

        # for i in range(3):
        #         for entry in extension_outer[:2,i,:].T:
        #                 for _ in entry:
        #                         if _ is not None:
        #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # plt.show()
        # for i in range(3):
        #         for entry in extension_double[:2,i,:].T:
        #                 for _ in entry:
        #                         if _ is not None:
        #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # plt.show()

        
        refresh=False

        depth_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        depth_middle  = sweep(phi0s, run, kw=naught_middle, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        depth_outer  = sweep(phi0s, run, kw=naught_outer, pre_process = final_depth, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        depth_double  = sweep(phi0s, run, kw=naught_double, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        refresh=False
        
        depth_middle_remodel  = sweep(phi0s, run, kw=naught_middle_remodel, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        depth_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = final_depth, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        depth_double_remodel  = sweep(phi0s, run, kw=naught_double_remodel, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # plt.plot(phi0s, depth_baseline, label='baseline')
        mid = depth_middle
        outer = depth_outer
        double = depth_double

        mid_remodel = depth_middle_remodel
        outer_remodel = depth_outer_remodel
        double_remodel = depth_double_remodel

        # plt.plot(phi0s, mid[:,0,:]-mid[:,idx,:], label='middle intercalations')
        # plt.plot(phi0s, outer[:,0,:]-outer[:,idx,:], label='outer intercalations')
        # plt.plot(phi0s, double[:,0,:]-double[:,idx,:], label='double intercalations')

        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(12, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')
        plt.sca(axs[0])
        for j in range(len(phi0s)):
                plt.plot(intercalations, mid[j,:]-depth_baseline[0], label=f'phi_0={phi0s[j]}')
                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(mid[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.max(mid[:,:,i])-mid_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
        # plt.legend()
        # plt.show()

        # fig, axs = plt.subplots(1,outer.shape[-1])
        # fig.set_size_inches(12, 4)
        # axs=axs.ravel()
        # plt.get_current_fig_manager().canvas.set_window_title('Outer (Depth)')
        plt.sca(axs[1])
        for j in range(len(phi0s)):
                p=plt.plot(intercalations, outer[j,:]-depth_baseline[0], label=f'phi_0={phi0s[j]}')
                plt.plot(intercalations, outer_remodel[j,:]-depth_baseline[0], color=p[0].get_color(), linestyle='-.', label=f'phi_0={phi0s[j]}')
        # for i in range(outer.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(outer[:,:,i]-depth_baseline))

                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(outer[:,:,i])-outer_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
        # plt.legend()
        # plt.show()

        # fig, axs = plt.subplots(1,double.shape[-1])
        # fig.set_size_inches(12, 4)
        # axs=axs.ravel()
        # plt.get_current_fig_manager().canvas.set_window_title('Double (Depth)')
        plt.sca(axs[2])
        for j in range(len(phi0s)):
                plt.plot(intercalations, double[j,:]-depth_baseline[0], label=f'phi_0={phi0s[j]}')
        # for i in range(double.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(double[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
        plt.legend()
        plt.show()



