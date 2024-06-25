import numpy as np
import matplotlib.pyplot as plt

from VertexTissue.Sweep import sweep
from VertexTissue.Geometry import unit_vector
from VertexTissue.Dict import dict_product
from VertexTissue.util import  pcolor

from Step1 import max_buckle_angle, run, phi0s, L0_T1s, base_path, middle_edge, outer_edge
from VertexTissue.Player import pickle_player

fontsize=14

    # main()
    
remodel = True
suffix = '' if remodel else '_no_remodel'


kws_outer = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':True})

kws_outer_belt = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':True, 'belt':True})

kws_outer_no_cable = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':True, 'cable':False})


kws_middle = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':False})

kws_middle_no_cable = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':False, 'cable':False})

kws_outer_no_remodel = dict_product({'L0_T1': L0_T1s, 'remodel': False, 'outer':True, 'cable':True})

kws_middle_no_cable_pit = dict_product({'L0_T1': L0_T1s, 'remodel': remodel, 'outer':False, 'cable':False, 'pit_before':True, 'pit_after':True})

kws_middle_inner_only = dict_product({'L0_T1': L0_T1s,  'outer':False, 'remodel':True, 'cable': True, 'belt': False, 'inner_only': True})
kws_middle_inner_only_no_remodel = dict_product({'L0_T1': L0_T1s,  'outer':False, 'remodel':False, 'cable': True, 'belt': False, 'inner_only': True})

kws_middle_triple = dict_product({'L0_T1': L0_T1s,  'outer':False, 'remodel':True, 'cable': True, 'belt': True, 'inner_only': False})

bar=(('outer', outer_edge),
        ('outer_no_cable', outer_edge),
        ('outer_belt', outer_edge),
        ('outer_no_remodel',outer_edge),
        ('middle', middle_edge),
        ('middle_no_cable', middle_edge),
        ('middle_no_cable_pit', middle_edge),
        ('middle_inner_only', middle_edge),
        ('middle_inner_only_no_remodel', middle_edge),
        ('middle_triple', middle_edge)
)

if __name__ == '__main__':




    local=locals()
    vmin = np.inf
    vmax = -np.inf
    angles={}

    for nm, edge in bar:
        kws=local['kws_'+nm]


        angles[nm] = (180/np.pi)*sweep(phi0s, run, kw=kws, pre_process = max_buckle_angle, pre_process_kw= {'edge': edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

        if np.any(np.isfinite(angles[nm])):
            vmin=min(vmin, np.nanmin(angles[nm]))
            vmax=max(vmax, np.nanmax(angles[nm]))


    vmin=0
    vmax=91
    for nm, _ in bar:
        plt.figure(num=nm);
        pcolor( L0_T1s , phi0s, angles[nm],  vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
        plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

        plt.tight_layout()

        plt.savefig(f'jackknifing_{nm}.pdf',dpi=200)
        plt.show()

    angle_outer=(180/np.pi)*sweep(phi0s, run, kw=kws_outer, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    angle_outer_no_cable=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_no_cable, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    angle_outer_belt=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_belt, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    angle_outer_no_remodel=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_no_remodel, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    angle_middle=(180/np.pi)*sweep(phi0s, run, kw=kws_middle, pre_process = max_buckle_angle, pre_process_kw= {'edge': middle_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    angle_middle_no_cable=(180/np.pi)*sweep(phi0s, run, kw=kws_middle_no_cable, pre_process = max_buckle_angle, pre_process_kw= {'edge': middle_edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

    vmin = min(angle_outer.min(), angle_middle.min())
    vmax = max(angle_outer.max(), angle_middle.max())

#     pcolor( L0_T1s , phi0s, angle_outer_belt)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_outer_belt.pdf',dpi=200)
#     plt.show()



#     pcolor( L0_T1s , phi0s, angle_outer, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_outer.pdf',dpi=200)
#     plt.show()

#     pcolor( L0_T1s , phi0s, angle_middle, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_middle.pdf',dpi=200)
#     plt.show()

#     pcolor( L0_T1s , phi0s, angle_middle_no_cable, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_middle_no_cable.pdf',dpi=200)
#     plt.show()

#     pcolor( L0_T1s , phi0s, angle_outer_no_remodel)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_outer_no_remodel.pdf',dpi=200)
#     plt.show()



#     pcolor( L0_T1s , phi0s, angle_outer_no_cable)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()

#     plt.savefig(f'jackknifing_outer_no_cable.pdf',dpi=200)
#     plt.show()