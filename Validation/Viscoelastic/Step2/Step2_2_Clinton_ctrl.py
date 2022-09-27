import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull



from VertexTissue.Sweep import sweep



from Step2_clinton import  final_depth, intercalations, run, phi0s,  base_path,    L0_T1s, clinton_middle, final_depth2


fontsize=14


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

        
        depth_middle  = sweep(phi0s, run, 
                              kw=clinton_middle,
                              pre_process = final_depth,
                              cache=True,
                              savepath_prefix=base_path,
                              inpaint=np.nan,
                              refresh=refresh,
                              pre_process_kw={'t_final':2e4})

        

        refresh=False
        # plt.plot(phi0s, depth_baseline, label='baseline')
        mid = depth_middle




        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(12.5, 4)
        # plt.get_current_fig_manager().canvas.set_window_title('Middle')
        axs=axs.ravel()
        # for i in range(mid.shape[-1]):
                # plt.sca(axs[i])
        plt.get_current_fig_manager().canvas.set_window_title('Depth (Basal)')

        plt.sca(axs[0])
        plt.title('Middle Region')
        plot(mid)
        plt.ylabel('$\Delta\;depth\;(\mu $m)')

        
        # for i in range(double.shape[-1]):
        #         plt.sca(axs[i])

                # pcolor( L0_T1s, list(reversed(phi0s)), np.flipud(double[:,:,i]-depth_baseline))
                # pcolor( L0_T1s, list(reversed(phi0s)), -np.flipud(np.nanmax(double[:,:,i])-double_remodel[:,:,i]))
                # plt.colorbar()
                # plt.show()
                            
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
        plt.tight_layout()   
        plt.savefig('clinton_invagination_depth_vs_intercalations.png',dpi=200)
        plt.show()


