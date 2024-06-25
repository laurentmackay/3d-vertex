import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull


from Validation.Viscoelastic.Step1.Step1 import buckle_angle_finder
from VertexTissue.Sweep import sweep
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.Dict import  last_dict_key, last_dict_value, take_dicts, dict_product
from VertexTissue.Memoization import get_caller_locals
from VertexTissue.Tissue import tissue_3d, get_outer_belt
from VertexTissue.util import pcolor

from Step2_bis import depth_timeline, intercalations, run, phi0s,  base_path, kws_baseline,  clinton_double, final_angle, L0_T1s, clinton_middle, clinton_outer, final_inter_arc_depth, final_depth, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline



remodel=True
L0_T1s=0
kws_baseline = dict_product({'intercalations':0, 'L0_T1':0.0, 'remodel':True})
kws_middle = dict_product({'intercalations':[4, 6, 18], 'outer':False,'double':False, 'remodel':remodel, 'L0_T1':L0_T1s})
kws_outer = dict_product({'intercalations':[4, 6, 18], 'outer':True, 'double': False, 'remodel':remodel, 'L0_T1':L0_T1s})
kws_double = dict_product({'intercalations':[4, 6, 18], 'outer':True,'double':True, 'remodel':remodel, 'L0_T1':L0_T1s})

fontsize=14

    # main()

_, G_apical = tissue_3d( hex=7,  basal=True)
belt = get_outer_belt(G_apical)
half_belt = belt[:int(len(belt)/2+1)]


if __name__ == '__main__':

        remodel=False
        L0_T1=L0_T1s
        idx=-1

        refresh=False


        

        timeline_func = extension_timeline
        
        
        depth_timeline_outer  = sweep(phi0s, run, kw=clinton_outer, pre_process = timeline_func, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=True, pass_kw=True)

        depth_timeline_outer_remodel  = sweep(phi0s, run, kw=naught_outer_remodel, pre_process = timeline_func, 
        cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, pass_kw=True)


        for entry, entry2, phi0 in zip(depth_timeline_outer_remodel, depth_timeline_outer, phi0s):
                if not np.isscalar(entry) and not entry[0] is None:
                        p=plt.plot(entry[0][:,0], entry[0][:,1], label=f'$\phi_0={phi0}$')
                        plt.plot(entry2[0][:,0], entry2[0][:,1],color=p[0].get_color(), linestyle='--')
                        for i in range(1,len(entry)):
                                if entry[i] is not None:
                                        plt.plot(entry[i][:,0], entry[i][:,1],color=p[0].get_color())
                                        plt.plot(entry2[i][:,0], entry2[i][:,1],color=p[0].get_color(), linestyle='--')

                                else:
                                        pass

                                    
        plt.legend()
        plt.show()