import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull



from VertexTissue.Sweep import sweep

from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask

from Step2_bis import angle_timeseries, depth_timeline, extension_timeseries, final_arc_ratio, final_cone_slope, final_depth, final_inter_arc_distance, final_width, inter_arc_distance_timeline, intercalations, run, phi0s, phi0_SLS,  base_path, kws_baseline,   final_angle, L0_T1s, final_inter_arc_depth, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline

from Step2_bis import kws_strong_pit_middle, kws_strong_pit_double, kws_strong_pit_outer, kws_strong_pit_baseline, kws_middle, kws_double, kws_outer, kws_middle_basal, kws_middle_basal_hi, kws_middle_fine, kws_middle_no_scale, kws_baseline_no_scale, kws_outer_no_scale, kws_middle_smolpit, kws_outer_smolpit, kws_baseline_smolpit
from Step2_bis import clinton_baseline, clinton_double, clinton_outer, clinton_middle, kws_baseline_thresh, kws_baseline_thresh_extend, kws_baseline_thresh_sym, ecs, kws_baseline_thresh_no_scale, kws_baseline_thresh_no_scale_extend, kws_baseline_thresh_no_scale_sym
from Step2_bis import kws_baseline_thresh_no_scale_no_T1, kws_baseline_thresh_no_scale_no_T1_extend, kws_baseline_thresh_no_scale_no_T1_sym, kws_baseline_thresh_no_scale_no_T1_edge, kws_baseline_thresh_no_scale_no_T1_extend_edge, kws_baseline_thresh_no_scale_no_T1_sym_edge, kws_SLS_baseline_thresh,  kws_SLS_baseline_thresh_ext,  kws_SLS_baseline_thresh_con
from VertexTissue.util import pcolor
from VertexTissue.visco_funcs import fluid_element
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


        refresh=True
      


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

        def plot(y):
                N=len(phi0s) 
                colors = plt.cm.nipy_spectral(np.linspace(0,.85,N))
                plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors )

                for j in range(len(phi0s)):
                        plt.plot(intercalations, y[j,:], label=f'$\phi_0={phi0s[j]}$', color=colors[j])      
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))

        def plot2(y1,y2):
                for j in range(len(phi0s)):
                        p=plt.plot(intercalations, y1[j,:], label=f'$\phi_0={phi0s[j]}$')
                        plt.plot(intercalations, y2[j,:], color=p[0].get_color(), linestyle='-.')
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))

        ec=0.4
        phi0=0.65
        crumpler = fluid_element(phi0=phi0, ec=ec)
        extender = fluid_element(phi0=phi0, ec=ec, contract=False, extend=True)
        sym = fluid_element(phi0=phi0, ec=ec, extend=True)

        lens = np.linspace(0,2,500)
        L0 = np.zeros(lens.shape)+1.0

        # plt.ion()

        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(9.5, 3.25)
        # # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()

        # import matplotlib as mpl
        # axs[0].set_ylabel(r'$\tau\dfrac{\dot{L}}{L}$', rotation = 0, fontsize=12)
        # crumpler = lens-1
        # crumpler[lens-1>-ec]=0
        # extender = lens-1
        # extender[lens-1<ec]=0
        # sym = lens-1
        # sym[np.logical_and(lens-1>-ec,lens-1<ec)]=0
        # for dLdt, ax, lbls, title in zip((crumpler, extender, sym), axs,
        #                              [((-ec,r'$-\varepsilon_{c}$'),),
        #                               ((ec,r'$\varepsilon_{c}$'),),
        #                               ((-ec,r'$-\varepsilon_{c}$'),(ec,r'$\varepsilon_{c}$'), )],
        #                                ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
        #         # dLdt = lens-1
        #         # dLdt[lens-1>-ec]=0
        #         plt.sca(ax)
        #         plt.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5)
        #         plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)     

        #         plt.plot(lens-1, lens-1, color='k', linestyle='--', linewidth=1, label='SLS')
        #         plt.plot(lens-1, dLdt, linewidth=2, label='Modified SLS')

        #         plt.xlim((-1,1))
        #         plt.ylim((-1,1))

        #         plt.draw()
        #         xticks=[-1.0, 0,  1.0]
        #         xlabels = [str(x) for x in xticks]
        #         print(xlabels[0])
        #         for lbl in lbls:
        #                 x=lbl[0]
        #                 lbl=lbl[1]
        #                 if x==-ec or x==ec:
        #                         plt.axvline(x=x, color='k', linestyle=':', linewidth=0.2)
        #                 if not np.any(xticks==x):
        #                        xticks.append(x)
        #                        xlabels.append(lbl)
        #                 else:
        #                         xlabels[np.argwhere(xticks==x)[0,0]]=lbl
        #         ax.set_xticks(xticks)
        #         ax.set_xticklabels(xlabels)
        #         ax.set_yticks((-1,0,1))
        #         ax.set_xlabel('Strain')
        #         ax.set_title(title)
        #         plt.legend(fontsize=8, loc='upper left')
        # plt.tight_layout()
        # # plt.show(block=True)
        # plt.savefig('SLS_elements_scheme.pdf')
        # plt.savefig('SLS_elements_scheme.png',dpi=200)
        
        # plt.show()
        # fig, axs = plt.subplots(1,3)
        # fig.set_size_inches(9.5, 3.25)
        # # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        # axs=axs.ravel()

        # import matplotlib as mpl
        # axs[0].set_ylabel('Equilibrium Length \n (non-dim)')
        # for l_rest, ax, lbls, title in zip((crumpler, extender, sym), axs,
        #                              [((-ec,r'$-\varepsilon_{c}$'),(-phi0,r'-$\phi_0$')),
        #                               ((ec,r'$\varepsilon_{c}$'),(phi0,r'$\phi_0$')),
        #                               ((-ec,r'$-\varepsilon_{c}$'),(ec,r'$\varepsilon_{c}$'), (-phi0,r'-$\phi_0$'), (phi0,r'$\phi_0$') )],
        #                                ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
        #         f_crumple = -(l_rest(lens,L0)-lens)
        #         plt.sca(ax)
        #         plt.axhline(y=1.0, color='k', linestyle='-', linewidth=0.5)
        #         plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)     

        #         plt.plot(lens-1, lens, color='k', linestyle='--', linewidth=1, label='Hookean Spring')
        #         plt.plot(f_crumple, lens, linewidth=2, label='Nonlinear Spring')

        #         plt.xlim((-1,1))
        #         plt.ylim((0,2))

        #         plt.draw()
        #         xticks=[-1.0, 0,  1.0]
        #         xlabels = [str(x) for x in xticks]
        #         print(xlabels[0])
        #         for lbl in lbls:
        #                 x=lbl[0]
        #                 lbl=lbl[1]
        #                 if x==-ec or x==ec:
        #                         plt.axvline(x=x, color='k', linestyle=':', linewidth=0.2)
        #                 if not np.any(xticks==x):
        #                        xticks.append(x)
        #                        xlabels.append(lbl)
        #                 else:
        #                         xlabels[np.argwhere(xticks==x)[0,0]]=lbl
        #         ax.set_xticks(xticks)
        #         ax.set_xticklabels(xlabels)
        #         ax.set_xlabel('Applied Force \n (non-dim)')
        #         ax.set_title(title)
        #         plt.legend(fontsize=8, loc='upper left')
        # plt.tight_layout()
        # # plt.show(block=True)
        # plt.savefig('VE_elements_scheme.pdf')
        # plt.savefig('VE_elements_scheme.png',dpi=200)

        depth_func = final_depth

        refresh=False
        
        kws_contract= kws_SLS_baseline_thresh_con
        kws_extend= kws_SLS_baseline_thresh_ext
        kws_sym= kws_SLS_baseline_thresh

        phi0s=phi0_SLS

        depth_baseline  = sweep(phi0s, run, kw=kws_contract, pre_process = depth_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)
        
        depth_extend  = sweep(phi0s, run, kw=kws_extend, pre_process = depth_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        depth_sym  = sweep(phi0s, run, kw=kws_sym, pre_process = depth_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        # depth_baseline_edge  = sweep(phi0s, run, kw=kws_baseline_thresh_no_scale_no_T1_edge, pre_process = depth_func,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)
        
        # depth_extend_edge = sweep(phi0s, run, kw=kws_baseline_thresh_no_scale_no_T1_extend_edge, pre_process = depth_func,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        # depth_sym_edge  = sweep(phi0s, run, kw=kws_baseline_thresh_no_scale_no_T1_sym_edge, pre_process = depth_func,
        # cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        
 
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(9.5, 3.5)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.setWindowTitle('Depth (Basal)')
        import matplotlib as mpl
        cmap = mpl.colors.Colormap('viridis')
        depths= (depth_baseline, depth_extend, depth_sym)
        vmax = max(*[np.nanmax(d) for d in depths])-depth_baseline[0,0]
        plt.sca(axs[0])
        plt.ylabel('$\delta$')
        for d, ax, title in zip(depths, axs,
                          ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
            plt.sca(ax)
            pcolor( ecs, 1-phi0s, d-depth_baseline[0,0], vmin=0, vmax=vmax)
            ax.set_title(title)
            plt.xlabel(r'$\varepsilon_c$')
        plt.colorbar()

        
        
        # plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout()   
        plt.savefig('invagination_depth_VE.pdf')
        plt.savefig('invagination_depth_VE.png', dpi=200)

        fig=plt.figure()
        fig.set_size_inches(6.5, 3.5)
        slice=6
        plt.plot(1-phi0s, depth_baseline[:,slice]-depth_baseline[0,0], color='c', label='Asymmetric: Contraction',linewidth=2)
        # plt.plot(phi0s, depth_baseline_edge[:,slice]-depth_baseline_edge[0,0], color='c', linestyle=':', label='Asymmetric: Contraction',linewidth=2)
        plt.plot(1-phi0s, depth_extend[:,slice]-depth_baseline[0,0], color='y',linewidth=2, label='Asymmetric: Extension')
        # plt.plot(phi0s, depth_extend_edge[:,slice]-depth_baseline_edge[0,0], color='y',linewidth=2, linestyle=':', label='Asymmetric: Extension')
        plt.plot(1-phi0s, depth_extend[:,slice]+depth_baseline[:,slice]-2*depth_baseline[0,0], color='g', linestyle='-.',linewidth=2, label='Asymmetric: Sum')
        # plt.plot(phi0s, depth_extend_edge[:,slice]+depth_baseline_edge[:,slice]-2*depth_baseline_edge[0,0], color='g', linestyle=':',linewidth=2, label='Asymmetric: Sum')
        plt.plot(1-phi0s, depth_sym[:,slice]-depth_baseline[0,0], color='m',linewidth=2, label='Symmetric')
        # plt.plot(phi0s, depth_sym_edge[:,slice]-depth_baseline_edge[0,0], color='m',linewidth=2, linestyle=':', label='Symmetric')
        plt.legend(loc='upper left')
        plt.ylabel('$\Delta\;depth\;(\mu $m)')
        plt.xlabel('$\delta$')
        plt.tight_layout()   
        # plt.savefig('asymmetric_vs_symmetric_depth_change.pdf')
        plt.savefig('asymmetric_vs_symmetric_depth_change.png', dpi=200)



        depth_timeline_sym = sweep(phi0s, run, kw=kws_sym, pre_process = depth_timeline,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)
        


        depth_timeline_ext = sweep(phi0s, run, kw=kws_extend, pre_process = depth_timeline,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)


        depth_timeline_con = sweep(phi0s, run, kw=kws_contract, pre_process = depth_timeline,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        plt.figure()
        for ts, phi0 in zip(depth_timeline_sym[:, slice], phi0s):
                if ts is not None:
                        plt.plot(ts[:,0], ts[:,1],label=f'\delta = {1-phi0}')

        plt.show()


        width_func = final_width

        refresh=False

        widths_baseline  = sweep(phi0s, run, kw=kws_contract, pre_process = width_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)
        
        widths_extend  = sweep(phi0s, run, kw=kws_extend, pre_process = width_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        widths_sym  = sweep(phi0s, run, kw=kws_sym, pre_process = width_func,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

        base=widths_baseline.reshape((-1,1))
 
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(9.5, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])

        import matplotlib as mpl
        plt.sca(axs[0])
        plt.ylabel('$\delta$')
        widths = ( widths_baseline, widths_extend, widths_sym)
        vmin = min(*[np.nanmin(d) for d in widths])-widths_baseline[0,0]
        print(vmin)
        for d,ax, title in zip(widths, axs, ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
            ax.set_title(title)
            plt.sca(ax)
            pcolor( ecs, phi0s, d-widths_baseline[0,0], vmin=vmin, vmax=0)
        #     plt.colorbar()
            plt.xlabel(r'$\varepsilon_c$')

        plt.colorbar()            
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout()   
        plt.savefig('invagination_widths_fluid.png', dpi=200)







        refresh=False
        width_ratio_baseline  = sweep(phi0s, run, kw=kws_contract, pre_process = final_cone_slope,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float, pre_process_kw={'arc':'outer'})
        
        width_ratio_extend  = sweep(phi0s, run, kw=kws_extend, pre_process = final_cone_slope,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float, pre_process_kw={'arc':'outer'})

        width_ratio_sym  = sweep(phi0s, run, kw=kws_sym, pre_process = final_cone_slope,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float, pre_process_kw={'arc':'outer'})

        base=width_ratio_baseline.reshape((-1,1))
 
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(9.5, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])

        import matplotlib as mpl
        ratios= (width_ratio_baseline, width_ratio_extend, width_ratio_sym)
        vmax = max(*[np.nanmax(d) for d in ratios])
        vmin = min(*[np.nanmin(d) for d in ratios])
        for d,ax in zip(ratios, axs):
            plt.sca(ax)
            pcolor( ecs, phi0s, d, vmin=vmin, vmax=vmax)
            plt.colorbar()

                            
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout()   
        # plt.savefig('invagination_widths_vs_intercalations.pdf')
        plt.show()


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