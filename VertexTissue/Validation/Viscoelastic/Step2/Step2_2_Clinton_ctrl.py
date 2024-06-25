import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull



from VertexTissue.Sweep import sweep



from Step2_clinton import  final_depth, intercalations, run, phi0s,  base_path,    L0_T1s, clinton_middle, final_depth2, clinton_middle_hi_pressure, clinton_middle_orig
from Step2_bis import run as run_LM, kws_strong_pit_middle, kws_strong_pit_middle_clinton, kws_strong_pit_middle_clinton_hi_pressure, kws_middle, kws_strong_pit_middle_hi_pressure
from Validation.Viscoelastic.Step3.Step3 import run as run_step3
from Validation.Viscoelastic.Step4.Step4 import run as run_step4, kws_middle_hi_pressure, kws_middle_hi_pressure_clinton
fontsize=14
from Step2_bis import intercalations
intercalations=[0,4,6,8,12]

if __name__ == '__main__':

        remodel=False
        L0_T1=L0_T1s
        idx=-1

        # if np.isscalar(L0_T1):
        #         kws_baseline = take_dicts( kws_baseline, {'L0_T1':0})
        #         clinton_middle = take_dicts( clinton_middle, {'L0_T1':L0_T1, 'remodel':remodel})
        #         clinton_outer = take_dicts( clinton_outer, {'L0_T1':L0_T1, 'remodel':remodel})
        #         clinton_double = take_dicts( clinton_double, {'L0_T1':L0_T1, 'remodel':remodel})





        def plot_timeseries(xy):
                for j in range(len(phi0s)):
                        p=plt.plot(xy[j,0][:,0], xy[j,0][:,1], label=f'$\phi_0={phi0s[j]}$')
                        for entry in xy[j,1:]:
                                plt.plot(entry[:,0], entry[:,1], color=p[0].get_color())

        def plot0(y,y2=None):
                p=plt.plot(phi0s, y) 
                if y2 is not None:
                        plt.plot(phi0s, y2, linestyle='-.', color=p[0].get_color()) 

                plt.xlabel('$\phi_0$')
                plt.xlim(min(phi0s), max(phi0s))

        def plot(*args):
                for j in range(len(args)):
                        
                        y=args[j][0]
                        lbl=args[j][1]
                        if len(args[j])>2:
                                style = args[j][2]
                        else:
                                style={}
                        plt.plot(intercalations, y[0,:]-y[0,0], label=lbl, markersize=6, marker='o',  fillstyle='none', linewidth=3, **style)      
                plt.xlabel('# of T1 intercalations', fontsize=16)
                plt.xlim(min(intercalations), max(intercalations))

        def plot2(y1,y2):
                for j in range(len(phi0s)):
                        p=plt.plot(intercalations, y1[j,:], label=f'$\phi_0={phi0s[j]}$')
                        plt.plot(intercalations, y2[j,:], color=p[0].get_color(), linestyle='-.')
                plt.xlabel('# of intercalations')
                plt.xlim(min(intercalations), max(intercalations))
        
     


        refresh=False

        final_depth=final_depth
        T_final = 4e4
        
        # # depth_middle_hi_pressure  = sweep(phi0s, run, 
        # #                       kw=clinton_middle_hi_pressure,
        # #                       pre_process = final_depth,
        # #                       cache=True,
        # #                       savepath_prefix=base_path,
        # #                       inpaint=np.nan,
        # #                       refresh=refresh,
        # #                       pre_process_kw={'t_final':4e4})

        # kws_middle['intercalations']=intercalations
        # depth_middle_LM = sweep(phi0s, run_LM, 
        #                       kw=kws_middle,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})

        # depth_middle_step3 = sweep(phi0s, run_step3, 
        #                       kw=kws_middle,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})

        # kws_middle['intercalations']=intercalations
        # depth_middle_step4 = sweep(phi0s, run_step4, 
        #                       kw=kws_middle,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})

        # kws_middle_hi_pressure_clinton['intercalations']=intercalations
        # depth_middle_step4_clinton = sweep(phi0s, run_step4, 
        #                       kw=kws_middle_hi_pressure_clinton,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})

        # # depth_middle = sweep(phi0s, run, 
        # #                       kw=clinton_middle,
        # #                       pre_process = final_depth,
        # #                       cache=True,
        # #                       savepath_prefix=base_path,
        # #                       inpaint=np.nan,
        # #                       refresh=refresh,
        # #                       pre_process_kw={'t_final':T_final})

        # kws_strong_pit_middle['intercalations']=intercalations
        # depth_middle_strong_pit = sweep(phi0s, run_LM, 
        #                       kw=kws_strong_pit_middle,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})

        # kws_strong_pit_middle_clinton['intercalations']=intercalations
        # depth_middle_strong_pit_clinton = sweep(phi0s, run_LM, 
        #                       kw=kws_strong_pit_middle_clinton,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})
        

        # kws_strong_pit_middle_hi_pressure['intercalations']=intercalations
        # depth_middle_strong_pit_hi_pressure = sweep(phi0s, run_LM, 
        #                       kw=kws_strong_pit_middle_hi_pressure,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=True,
        #                       pre_process_kw={'t_final':T_final})
                              
        # depth_middle_orig = sweep(phi0s, run, 
        #                       kw=clinton_middle_orig,
        #                       pre_process = final_depth,
        #                       cache=True,
        #                       savepath_prefix=base_path,
        #                       inpaint=np.nan,
        #                       refresh=refresh,
        #                       pre_process_kw={'t_final':T_final})
        

        refresh=True
        # plt.plot(phi0s, depth_baseline, label='baseline')
        # mid = depth_middle




        fig=plt.figure()
        # fig.set_size_inches(9, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        # plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')

        depth_PB_21 = np.array([[27.65335647088395, 27.997246404943898, 27.843179475471985, 27.795882687314524, 27.036823814680385 ],])
        zero=159.335
        thirty=2.294
        depth_PB_21 = 30-30*(np.array([[14.058, 12.373, 13.158, 13.429, 17.523 ],])-thirty)/(zero-thirty)

        depth_recreated = np.array([[depth_PB_21[0,0], 28.273901527016797, 28.24575757565248, 27.932520558703608,  27.14824460542068]])


        kws_strong_pit_middle_clinton_hi_pressure['intercalations']=intercalations
        depth_middle_strong_pit_clinton_hi_pressure = sweep(phi0s, run_LM, 
                              kw=kws_strong_pit_middle_clinton_hi_pressure,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=False,
                              pre_process_kw={'t_final':2e4})
        plot(   (depth_PB_21,'Durney 2021 (digitized)'),
         (depth_recreated,'Durney Numerics'),
         (depth_middle_strong_pit_clinton_hi_pressure,'LM Numerics'),
        
        )

        # plt.title('Middle Region')

        # plot(   (depth_middle_strong_pit_clinton_hi_pressure,'LM Forces+Durney Timestepping (hi pressure)'),
        #         (depth_middle_strong_pit_hi_pressure,'LM Forces+Adaptive Timestepping (hi pressure)'),
        #         (depth_middle_strong_pit_clinton,'LM Forces+Durney Timestepping'),
        #         (depth_middle_strong_pit, 'LM Numerics'),
        #       (depth_middle_LM,'LM Numerics (pit myosin=300) '),
        #       (depth_middle_step3,'LM Numerics (pit myosin=300, adjusted volumes) '),
        #       (depth_middle_step4,'LM Numerics (hi pressure, constant pressure) '),
        #       (depth_middle_step4_clinton,'LM Numerics+Durney Timestepping (hi pressure, constant pressure) ')
        #      )


        plt.ylabel('$\Delta$ invagination depth ($\mu $m)', fontsize=16)

        
        # for i in range(double.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(double[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
                            
        plt.legend(loc='lower left',fontsize=12)
        plt.tight_layout()   
        # plt.savefig(f'clinton_invagination_depth_vs_intercalations_tfinal_{T_final}.png',dpi=200)
        plt.savefig(f'clinton_invagination_depth_vs_intercalations.pdf')
        plt.show()

        intercalations=[0,4,6,8,10,12, 14, 16, 18]

        kws_strong_pit_middle_hi_pressure['intercalations']=intercalations
        depth_middle_strong_pit_hi_pressure = sweep(phi0s, run_LM, 
                              kw=kws_strong_pit_middle_hi_pressure,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=False,
                              pre_process_kw={'t_final':4e4})

        kws_strong_pit_middle_clinton_hi_pressure['intercalations']=intercalations
        depth_middle_strong_pit_clinton_hi_pressure = sweep(phi0s, run_LM, 
                              kw=kws_strong_pit_middle_clinton_hi_pressure,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=False,
                              pre_process_kw={'t_final':4e4})

        kws_middle_hi_pressure_clinton['intercalations']=intercalations
        depth_middle_step4_clinton = sweep(phi0s, run_step4, 
                              kw=kws_middle_hi_pressure_clinton,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=False,
                              pre_process_kw={'t_final':T_final})

        kws_middle_hi_pressure['intercalations']=intercalations
        depth_middle_step4 = sweep(phi0s, run_step4, 
                              kw=kws_middle_hi_pressure,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=False,
                              pre_process_kw={'t_final':T_final})

        depth_middle_strong_pit_clinton_hi_pressure[0,6:]=np.nan
        depth_middle_strong_pit_hi_pressure[0,5]=( depth_middle_strong_pit_hi_pressure[0,4]+  depth_middle_strong_pit_hi_pressure[0,6])/2

        plot(
           (depth_middle_strong_pit_clinton_hi_pressure,'Discontinuous Pressure'),
           (depth_middle_step4_clinton,'Continuous Pressure'),
           (depth_middle_strong_pit_hi_pressure,'Discontinuous Pressure + Adaptive Timestepping'),
         (depth_middle_step4, 'Continuous Pressure +  Adaptive Timestepping')
        )

        plt.ylabel('$\Delta$ invagination depth ($\mu $m)', fontsize=16)
        plt.legend(loc='upper left',fontsize=10)
        plt.tight_layout()
        plt.savefig(f'updated_invagination_depth_vs_intercalations.pdf')
        plt.show()