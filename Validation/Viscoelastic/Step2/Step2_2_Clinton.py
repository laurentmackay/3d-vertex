import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull



from VertexTissue.Sweep import sweep

from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask

from Step2_bis import angle_timeseries, depth_timeline, extension_timeseries, final_arc_ratio, final_depth, final_inter_arc_distance, final_width, inter_arc_distance_timeline, intercalations, run, phi0s, phi0_SLS,  base_path, kws_baseline,   final_angle, L0_T1s, final_inter_arc_depth, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline

from Step2_bis import kws_strong_pit_middle, kws_strong_pit_double, kws_strong_pit_outer, kws_strong_pit_baseline, kws_middle, kws_double, kws_outer, kws_middle_basal, kws_middle_basal_hi, kws_middle_fine, kws_middle_no_scale, kws_baseline_no_scale, kws_outer_no_scale, kws_middle_smolpit, kws_outer_smolpit, kws_baseline_smolpit
from Step2_bis import clinton_baseline, clinton_double, clinton_outer, clinton_middle, kws_inter_thresh_no_scale, kws_inter_thresh_no_scale_ext, kws_inter_thresh_no_scale_sym, kws_baseline_thresh_no_scale_slice, kws_baseline_thresh_no_scale_slice_sym, kws_baseline_thresh_no_scale_slice_ext
from Step2_bis import kws_SLS_baseline, kws_SLS_middle, kws_SLS_outer, kws_SLS_baseline_ext, kws_SLS_middle_ext, kws_SLS_outer_ext, kws_SLS_baseline_con, kws_SLS_middle_con, kws_SLS_outer_con
fontsize=14


if __name__ == '__main__':

        remodel=False
        L0_T1=L0_T1s
        idx=-1

        # if np.isscalar(L0_T1):
        #         kws_strong_pit_baseline = take_dicts( kws_strong_pit_baseline, {'L0_T1':0})
        #         clinton_middle = take_dicts( clinton_middle, {'L0_T1':L0_T1, 'remodel':remodel})
        #         clinton_outer = take_dicts( clinton_outer, {'L0_T1':L0_T1, 'remodel':remodel})
        #         clinton_double = take_dicts( clinton_double, {'L0_T1':L0_T1, 'remodel':remodel})


        # clinton_middle_remodel = dict_product({'intercalations':intercalations, 'remodel':True, 'L0_T1':0})
        # clinton_outer_remodel = dict_product({'intercalations':intercalations, 'outer':True, 'remodel':True, 'L0_T1':0})
        # clinton_double_remodel = dict_product({'intercalations':intercalations, 'outer':True,'double':True, 'remodel':True, 'L0_T1':0})


        refresh=False
        
        N=12
        kws_middle_slice = kws_middle.copy()
        kws_middle_slice['intercalations']=N

        kws_outer_slice = kws_outer.copy()
        kws_outer_slice['intercalations']=N

        angle_middle  = sweep(phi0s, run, kw=kws_middle_slice, pre_process = angle_timeseries, pass_kw=True, pre_process_kw={'basal': False, 'summary':np.mean},
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        angle_outer  = sweep(phi0s, run, kw=kws_outer_slice, pre_process = angle_timeseries, pass_kw=True, pre_process_kw={'basal': False, 'summary':np.mean},
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        # angle_double  = (180/np.pi)*sweep(phi0s, run, kw=clinton_double, pre_process = final_angle, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        # for j in range(len(phi0s)):
        #         plt.plot(intercalations, angle_middle[j,:], label=f'phi_0={phi0s[j]}')
        # plt.legend()
        # plt.show()

        # for j in range(len(phi0s)):
        #         plt.plot(intercalations, angle_outer[j,:], label=f'phi_0={phi0s[j]}')
        # plt.legend() 

        
        # plt.show()

        # refresh = False
        # extension_middle  = sweep(phi0s, run, kw=clinton_middle, pre_process = inter_arc_distance_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # extension_outer  = sweep(phi0s, run, kw=clinton_outer, pre_process = inter_arc_distance_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # extension_double  = sweep(phi0s, run, kw=clinton_double, pre_process = inter_arc_distance_timeline, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        def plot_timeseries(xy):
                for j in range(len(phi0s)):
                        p=plt.plot(xy[j][:,0], xy[j][:,1], label=f'$\phi_0={phi0s[j]}$')


        def plot0(y,y2=None):
                p=plt.plot(phi0s, y) 
                if y2 is not None:
                        plt.plot(phi0s, y2, linestyle='-.', color=p[0].get_color()) 

                plt.xlabel('$\phi_0$')
                plt.xlim(min(phi0s), max(phi0s))

        def plot(y, label=True):
                N=len(phi0s) 
                colors = plt.cm.nipy_spectral(np.linspace(0,.85,N))
                plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors )

                for j in range(len(phi0s)):
                        plt.plot(intercalations, y[j,:], label=f'$\delta={phi0s[j]}$' if label else None, color=colors[j])      
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))

        def plot2(y1,y2):
                for j in range(len(phi0s)):
                        p=plt.plot(intercalations, y1[j,:], label=f'$\phi_0={phi0s[j]}$')
                        plt.plot(intercalations, y2[j,:], color=p[0].get_color(), linestyle='-.')
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))


        # plot_timeseries(angle_middle)
        # plt.show()

        # plot_timeseries(angle_outer)
        # plt.show()
        
        # fig, axs = plt.subplots(1,3)
        # axs=axs.ravel()
        # fig.set_size_inches(12, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Extension')
        # plt.sca(axs[0])
        # plot_timeseries(extension_middle)


        # # plt.get_current_fig_manager().canvas.set_window_title('Outer (Tissue Extension)')

        # plt.sca(axs[1])
        # plot_timeseries(extension_outer)
        # # plt.legend() 
        # # plt.show()


        # # plt.get_current_fig_manager().canvas.set_window_title('Double (Tissue Extension)')
        # plt.sca(axs[2])
        # plot_timeseries(extension_double)

        # plt.legend() 
        # plt.show()

        # refresh=False
        # colors=['r','g','b']

        # extension_func = final_inter_arc_distance
        # extension_middle  = sweep(phi0s, run, kw=clinton_middle, pre_process = extension_func, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # extension_outer  = sweep(phi0s, run, kw=clinton_outer, pre_process = extension_func, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # extension_double  = sweep(phi0s, run, kw=clinton_double, pre_process = extension_func, pass_kw=True,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)  

        # # extension_middle_remodel  = sweep(phi0s, run, kw=naught_middle_remodel, pre_process = extension_func, pass_kw=False,
        # # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # # extension_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = extension_func, pass_kw=False,
        # # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # # extension_double_remodel  = sweep(phi0s, run, kw=naught_double_remodel, pre_process = extension_func, pass_kw=False,
        # # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)  

        # fig, axs = plt.subplots(1,3)
        # axs=axs.ravel()
        # fig.set_size_inches(15.5, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Extension')
        # plt.sca(axs[0])
        # plt.title('Middle Region')
        # plot0((extension_middle-extension_middle[0,0])[:,-1])

        # # plt.xlim(min(intercalations), max(intercalations))
        # plt.ylabel(r'$\Delta\;extension\;(\mu $m)')
        # # plt.xlabel('# of intercalations')
        # # plt.get_current_fig_manager().canvas.set_window_title('Outer (Tissue Extension)')

        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot0((extension_outer-extension_outer[0,0])[:,-1])

        # # plt.xlim(min(intercalations), max(intercalations))
        # plt.ylabel(r'$\Delta\;extension\;(\mu $m)')
        # # plt.xlabel('# of intercalations')

        # # plt.get_current_fig_manager().canvas.set_window_title('Double (Tissue Extension)')

        # plt.sca(axs[2])
        # plt.title('Both')
        # # plot2(extension_double-extension_double[0,0], (extension_middle-extension_middle[0,0])+(extension_outer-extension_outer[0,0]))
        # plot0((extension_double-extension_double[0,0])[:,-1], (extension_outer-extension_outer[0,0]+extension_middle-extension_middle[0,0])[:,-1])
        # # plt.xlim(min(intercalations), max(intercalations))
        # plt.ylabel(r'$\Delta\;extension\;(\mu $m)')
        # # plt.xlabel('# of intercalations')

        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975)) 
        # plt.savefig('extension_vs_phi0.png',dpi=200)
        # plt.show()
        # # for i in range(3):
        # #         for entry in extension_middle[:2,i,:].T:
        # #                 for _ in entry:
        # #                         if _ is not None:
        # #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # # plt.show()

        # # for i in range(3):
        # #         for entry in extension_outer[:2,i,:].T:
        # #                 for _ in entry:
        # #                         if _ is not None:
        # #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # # plt.show()
        # # for i in range(3):
        # #         for entry in extension_double[:2,i,:].T:
        # #                 for _ in entry:
        # #                         if _ is not None:
        # #                                 plt.plot(_[:,0],_[:,1], color=colors[i])

        # # plt.show()


        depth_func = final_depth

        refresh=False
        overwrite=False
        # phi0s=phi0s[phi0s>=0.5]
        # kws_baseline = kws_SLS_baseline

        # kws_middle = kws_SLS_middle
        # kws_outer = kws_SLS_outer
        phi0s=phi0_SLS

        fig, axs = plt.subplots(3,2)
        fig.set_size_inches(9.5, 8)
        axs=axs.ravel()

        for i, kws in enumerate(((kws_SLS_baseline_con, kws_SLS_middle_con, kws_SLS_outer_con),
                                (kws_SLS_baseline_ext,kws_SLS_middle_ext, kws_SLS_outer_ext),
                                (kws_SLS_baseline,kws_SLS_middle, kws_SLS_outer))):
                
                kws_baseline, kws_middle, kws_outer  = kws
                
                depth_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = depth_func,
                                        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, overwrite=overwrite)

        
                depth_middle  = sweep(phi0s, run, kw=kws_middle, pre_process = depth_func,
                                        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
                

                depth_outer  = sweep(phi0s, run, kw=kws_outer, pre_process = depth_func,
                        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, overwrite=overwrite)
                
                base=depth_baseline.reshape((-1,1))
                mid = depth_middle
                outer = depth_outer


                plt.sca(axs[i*2])
                plt.title('Middle Region')
                plt.plot([0,18],[0,0],linestyle='--', color='k')
                plot(mid-base, label=False)
                plt.ylabel('$\Delta\;depth\;(\mu $m)')
                plt.xticks(ticks=[0,5,10,15])

                plt.sca(axs[(i)*2+1])
                plt.title('Outer Region')
                # plt.plot(intercalations, (depth_outer_step3-depth_baseline[0]).T, linestyle='--',color=[0,0,0])
                plt.plot([0,18],[0,0],linestyle='--', color='k')
                plot(outer-base, label=i==2)
                # plt.ylim((0,2.85))
                plt.xticks(ticks=[0,5,10,15])

                
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        plt.tight_layout() 
        fig.legend(lines, labels,   loc='outside center right', bbox_to_anchor=(1.05, 0.5),
           bbox_transform = plt.gcf().transFigure)
          
        # plt.savefig('invagination_depth_vs_intercalations.pdf')
        plt.savefig('invagination_depth_vs_intercalations.png',dpi=200)
        plt.show()

        # phi0s=phi0s[phi0s>0.1]



        from Validation.Viscoelastic.Step4.Step4 import run as run_step3, kws_middle as kws_middle_step3, intercalations as inter3, kws_outer as kws_outer_step3
        
        kws_middle_step3['intercalations']=intercalations
        depth_middle_step3  = sweep([1.0,], run_step3, kw=kws_middle_step3, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # depth_middle_basal  = sweep([1.0,], run, kw=kws_middle_fine, pre_process = depth_func ,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=True)




        kws_outer_step3['intercalations']=intercalations
        depth_outer_step3  = sweep([1.0,], run_step3, kw=kws_outer_step3, pre_process = depth_func ,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # depth_double  = sweep(phi0s, run, kw=kws_double, pre_process = depth_func,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=False)

        refresh=False
        
        # depth_middle_remodel  = sweep(phi0s, run, kw=naught_middle_remodel, pre_process = final_depth2,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # depth_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = final_depth2, 
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # depth_double_remodel  = sweep(phi0s, run, kw=naught_double_remodel, pre_process = final_depth2,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # plt.plot(phi0s, depth_baseline, label='baseline')
        

        # base[:]=0;

        # double = depth_double

        # mid_remodel = depth_middle_remodel
        # outer_remodel = depth_outer_remodel
        # double_remodel = depth_double_remodel


        plt.plot(phi0s, base)



        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.setWindowTitle('Depth (Basal)')

        # plt.sca(axs[0])
        # N=len(intercalations) 
        # colors = plt.cm.nipy_spectral(np.linspace(0,.85,N))
        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors )

        # for j in range(len(intercalations)):
        #         plt.plot(phi0s, depth_middle[:,j], label=f'$\intercalations={intercalations[j]}$', color=colors[j])   
        # # plt.plot(phi0s, )
        # plt.ylabel('$depth\;(\mu $m)')
        # # plt.xlabel(r'$\phi_0$')


        




        #plt.ylabel('$\Delta\;depth\;(\mu $m)')

        # plt.sca(axs[2])
        # plt.title('Both')
        # # plot2(double-depth_baseline[0],mid+outer-2*depth_baseline[0])
        # plot(double-base)
        #plt.ylabel('$\Delta\;depth\;(\mu $m)')

        
        # for i in range(double.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(double[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
                            




        # refresh=False


        # width_ratio_baseline  = sweep(phi0s, run, kw=kws_strong_pit_baseline, pre_process = final_arc_ratio,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        # width_ratio_middle  = sweep(phi0s, run, kw=clinton_middle, pre_process = final_arc_ratio,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # width_ratio_outer  = sweep(phi0s, run, kw=clinton_outer, pre_process = final_arc_ratio,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        # width_ratio_double  = sweep(phi0s, run, kw=clinton_double, pre_process = final_arc_ratio,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)




        # mid = width_ratio_middle
        # outer = width_ratio_outer
        # double = width_ratio_double




        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(12.5, 4)
        # # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()
        # # for i in range(mid.shape[-1]):
        #         # plt.sca(axs[i])
        # plt.get_current_fig_manager().canvas.set_window_title('Width Ratio')

        # plt.sca(axs[0])
        # plt.title('Middle Region')
        # plot(mid)
        # plt.ylabel('Width Ratio')

        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot(outer)
        # #plt.ylabel('$\Delta\;depth\;(\mu $m)')

        # plt.sca(axs[2])
        # plt.title('Both')
        # # plot2(double-depth_baseline[0],mid+outer-2*depth_baseline[0])
        # plot(double)

        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        # plt.savefig('width_ratio_vs_intercalations.png',dpi=200)
        # plt.tight_layout()  
        # plt.show()

        # refresh=False


        # width_baseline  = sweep(phi0s, run, kw=kws_strong_pit_baseline, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)
        
        # width_middle  = sweep(phi0s, run, kw=clinton_middle, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # width_outer  = sweep(phi0s, run, kw=clinton_outer, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        

        # width_double  = sweep(phi0s, run, kw=clinton_double, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # refresh=False
        
        # width_middle_remodel  = sweep(phi0s, run, kw=naught_middle_remodel, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # width_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = final_width, 
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        # width_double_remodel  = sweep(phi0s, run, kw=naught_double_remodel, pre_process = final_width,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        # # plt.plot(phi0s, depth_baseline, label='baseline')
        # mid = width_middle
        # outer = width_outer
        # double = width_double

        # mid_remodel = width_middle_remodel
        # outer_remodel = width_outer_remodel
        # double_remodel = width_double_remodel


        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(12.5, 4)
        # # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()
        # # for i in range(mid.shape[-1]):
        #         # plt.sca(axs[i])
        # plt.get_current_fig_manager().canvas.set_window_title('Width (Basal)')

        # plt.sca(axs[0])
        # plt.title('Middle Region')
        # plot(mid-width_baseline[0])
        # plt.ylabel('$\Delta\;width\;(\mu $m)')

        # plt.sca(axs[1])
        # plt.title('Outer Region')
        # plot(outer-width_baseline[0])
        # #plt.ylabel('$\Delta\;width\;(\mu $m)')

        # plt.sca(axs[2])
        # plt.title('Both')
        # # plot2(double-width_baseline[0],mid+outer-2*width_baseline[0])
        # plot(double-width_baseline[0])
        # #plt.ylabel('$\Delta\;width\;(\mu $m)')

        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        # plt.tight_layout()   
        # plt.savefig('invagination_width_vs_intercalations.png',dpi=200)
        # plt.show()