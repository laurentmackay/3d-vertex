import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


from Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder
from VertexTissue.Sweep import sweep
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.Dict import  last_dict_key, last_dict_value, take_dicts
from VertexTissue.Memoization import get_caller_locals
from VertexTissue.Tissue import tissue_3d, get_outer_belt

from Step2_bis import  run, phi0s,  base_path, kws, kws_baseline, get_inter_edges, lumen_depth, width_timeline, final_width, final_depth, depth_timeline
from VertexTissue.Player import pickle_player

fontsize=14

    # main()

_, G_apical = tissue_3d( hex=7,  basal=True)
belt = get_outer_belt(G_apical)
half_belt = belt[:int(len(belt)/2+1)]







def extension(G, intercalations=0, outer=False, double=False, **kw):
        inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

        lens = [euclidean_distance(G.nodes[e[0]]['pos'], G.nodes[e[1]]['pos']) for e in inter_edges]

        return max(lens)


def angle(G, intercalations=0, outer=False, double=False, **kw):
        inter_edges = get_inter_edges(intercalations=intercalations, outer=outer, double=double)

        lens = [  buckle_angle_finder(G, edge=e)(G) for e in inter_edges]


        return max(lens)

if __name__ == '__main__':

        # kws_baseline = take_dicts( kws_baseline, {'L0_T1':0, 'remodel':True})
        # kws_middle = take_dicts( kws_middle, {'L0_T1':0, 'remodel':True})
        # kws_outer = take_dicts( kws_outer, {'L0_T1':0, 'remodel':True})
        # kws_double = take_dicts( kws_double, {'L0_T1':0, 'remodel':True})




        
        def final_time_check(d, **kw):
                if last_dict_key(d)<3.99e4:
                        print(get_caller_locals()['path'], end=' ')




                        


                return lumen_depth()

        
        def final_extension(d, **kw):
                return extension(last_dict_value(d), **kw)

        def final_angle(d, **kw):
                return angle(last_dict_value(d), **kw)


        refresh=False

        width_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = final_width,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)

        width_timelines  = sweep(phi0s, run, kw=kws_baseline, pre_process = width_timeline,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        for entry, phi0 in zip(width_timelines, phi0s):
                if not np.isscalar(entry) and not entry is None:
                        plt.plot(entry[:,0], entry[:,1], label=f'$\phi_0={phi0}$')

        plt.xlim([0, 40000])
        plt.legend()
        
        plt.ylabel(r'Lumen Width $(\mu {\rm m})$',fontsize=fontsize)
        plt.xlabel(r'Time $({\rm s})$',fontsize=fontsize)
        plt.savefig(f'VE_invagination_width_timeline.pdf',dpi=200)
        plt.show()



        



        plt.plot(phi0s, width_baseline)
        plt.xlabel(r'$\phi_{0}$',fontsize=fontsize)
        plt.ylabel(r'Final Width $(\mu {\rm m})$',fontsize=fontsize)

        plt.savefig(f'VE_invagination_width.pdf',dpi=200)
        # plt.legend()
        plt.show()

        # refresh=False



        depth_baseline  = sweep(phi0s, run, kw=kws_baseline, pre_process = final_depth,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)



        plt.plot(phi0s, depth_baseline,label='baseline')
        plt.xlabel(r'$\phi_{0}$',fontsize=fontsize)
        plt.ylabel(r'Final Depth $(\mu {\rm m})$',fontsize=fontsize)

        plt.ylim(np.nanmean(depth_baseline)+np.array([-1,1])*(np.nanmax(width_baseline)-np.nanmin(width_baseline))/2)

        # plt.legend()
        plt.savefig(f'VE_invagination_depth.pdf',dpi=200)
        plt.show()

    

        depth_timelines  = sweep(phi0s, run, kw=kws_baseline, pre_process = depth_timeline,
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh)


        for entry, phi0 in zip(depth_timelines, phi0s):
                if not np.isscalar(entry) and not entry is None:
                        plt.plot(entry[:,0], entry[:,1], label=f'$\phi_0={phi0}$')
                else:
                        pass

        plt.xlim([0, 40000])
        plt.legend()
        
        plt.ylabel(r'Lumen Depth $(\mu {\rm m})$',fontsize=fontsize)
        plt.xlabel(r'Time $({\rm s})$',fontsize=fontsize)
        plt.savefig(f'VE_invagination_depth_timeline.pdf',dpi=200)
        plt.show()