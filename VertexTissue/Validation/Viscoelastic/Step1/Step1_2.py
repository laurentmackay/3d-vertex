import numpy as np
import matplotlib.pyplot as plt

from VertexTissue.Analysis import sweep, parameter_keywords_sweep_analyzer
from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.util import dict_product, hatched_contour, last_dict_value, script_name, pcolor, upsample, contour

from Step1 import extending_edge_length, max_buckle_angle, run, phi0s, L0_T1s, base_path, middle_edge, outer_edge
import Step1_1
from Step1_1 import  bar
from VertexTissue.Player import pickle_player





def extension_timeseries(d, edge=None):
    return np.array([extending_edge_length(G, edge=edge) for G in d.values()])
    

def foo(d, edge=None):
    # t=np.array([*d.keys()])
    ell = extension_timeseries(d, edge=edge)
    # plt.plot(t,ell)
    # plt.show()
    return ell[-1]

fontsize=14

if __name__ == '__main__':

        
    contour_color = 'c'

    buckle_thresh=12

    local=locals()
    vmin = np.inf
    vmax = -np.inf
    lens={}
    angles={}
    intrinsic_stretch={}
    forced_stretch={}
    for nm, edge in bar:
        kws=Step1_1.__dict__['kws_'+nm]

        lens[nm]=sweep(phi0s, run, kw=kws, pre_process=foo, pre_process_kw={'edge': edge}, 
                  cache=True,  savepath_prefix=base_path, inpaint=np.nan)

        angles[nm] = (180/np.pi)*sweep(phi0s, run, kw=kws, pre_process = max_buckle_angle, pre_process_kw= {'edge': edge},
            cache=True, savepath_prefix=base_path, inpaint=np.nan)

        intrinsic_stretch[nm]=lens[nm][:,0]
        forced_stretch[nm]=lens[nm][:,-1]

        if np.any(np.isfinite(lens[nm])):
            vmin=min(vmin, np.nanmin(lens[nm]))
            vmax=max(vmax, np.nanmax(lens[nm]))

    
    for nm, _ in bar:
        plt.figure(num=nm);
        pcolor(L0_T1s, phi0s, lens[nm], vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
        plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)
        
        if np.any(angles[nm]>buckle_thresh):
            contour(L0_T1s, phi0s, angles[nm], upscale=100, levels=[buckle_thresh,135])

        plt.tight_layout()
        plt.savefig(f'final_extension_{nm}.pdf',dpi=200)
        # plt.show()

    lbls={'middle_no_cable':'Middle',
        'middle':'Middle+Arcs',
        'middle_triple':'Middle+Arcs+Belt' ,
        'middle_no_cable_pit':'Middle+Pit' ,
        'middle_inner_only':'Middle+Inner Arc',
        'middle_inner_only_no_remodel':r'Middle+Inner Arc+$\tau=\infty$',
    'outer_no_cable': 'Outer',
    'outer':'Outer+Arcs',
    'outer_no_remodel':r'Outer+Arcs+$\tau=\infty$',
    'outer_belt':'Outer+Arcs+Belt',  
          }

    plt.figure(figsize=(8, 4)).clear()
    for nm in sorted(lbls.keys(), key=lambda nm: np.nanmean(intrinsic_stretch[nm]), reverse=True):
        if nm in intrinsic_stretch.keys():
            plt.plot(phi0s, intrinsic_stretch[nm],label=lbls[nm])

    plt.legend(loc='center left',ncol=1, fancybox=True, bbox_to_anchor=(1.05, 0.5)) 
    plt.xlabel(r'$\phi_{0}$',fontsize=fontsize)
    plt.ylabel(r'$||\vec{bc}||$ ($\mu$m)',fontsize=fontsize)
    plt.xlim(phi0s[0], phi0s[-1])
    plt.tight_layout()

    plt.savefig(f'final_extension_summary.pdf',dpi=200)


    plt.figure(figsize=(8, 4)).clear()
    for nm in sorted(lbls.keys(), key=lambda nm: np.nanmean(intrinsic_stretch[nm]), reverse=True):
        if nm in intrinsic_stretch.keys():
            plt.plot(phi0s, forced_stretch[nm],label=lbls[nm])

    plt.legend(loc='center left',ncol=1, fancybox=True, bbox_to_anchor=(1.05, 0.5)) 
    plt.xlabel(r'$\phi_{0}$',fontsize=fontsize)
    plt.ylabel(r'$||\vec{bc}||$ ($\mu$m)',fontsize=fontsize)
    plt.xlim(phi0s[0], phi0s[-1])
    plt.tight_layout()

    plt.savefig(f'final_forced_extension_summary.pdf',dpi=200)
       
    plt.show()

    # lens_outer=sweep(phi0s, run, kw=kws_outer, pre_process=foo, pre_process_kw={'edge': outer_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)


    # lens_outer_belt=sweep(phi0s, run, kw=kws_outer_belt, pre_process=foo, pre_process_kw={'edge': outer_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    # lens_outer_no_cable=sweep(phi0s, run, kw=kws_outer_no_cable, pre_process=foo, pre_process_kw={'edge': outer_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    # lens_outer_no_remodel=sweep(phi0s, run, kw=kws_outer_no_remodel, pre_process=foo, pre_process_kw={'edge': outer_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    # lens_middle=sweep(phi0s, run, kw=kws_middle, pre_process=foo, pre_process_kw={'edge': middle_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    # lens_middle_no_cable=sweep(phi0s, run, kw=kws_middle_no_cable, pre_process=foo, pre_process_kw={'edge': middle_edge}, 
    #               cache=True,  savepath_prefix=base_path, inpaint=np.nan)

    # angle_outer=(180/np.pi)*sweep(phi0s, run, kw=kws_outer, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)

    # angle_outer_belt=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_belt, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)

    # angle_outer_no_cable=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_no_cable, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)

    # angle_outer_no_remodel=(180/np.pi)*sweep(phi0s, run, kw=kws_outer_no_remodel, pre_process = max_buckle_angle, pre_process_kw= {'edge': outer_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)
    
    # angle_middle=(180/np.pi)*sweep(phi0s, run, kw=kws_middle, pre_process = max_buckle_angle, pre_process_kw= {'edge': middle_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)

    # angle_middle_no_cable=(180/np.pi)*sweep(phi0s, run, kw=kws_middle_no_cable, pre_process = max_buckle_angle, pre_process_kw= {'edge': middle_edge},
    #         cache=True, savepath_prefix=base_path, inpaint=np.nan)
    
    # vmin=min(*[np.nanmin(l) for l in (lens_middle, lens_outer, lens_outer_belt)])
    # vmax=max(*[np.nanmax(l) for l in (lens_middle, lens_outer, lens_outer_belt)])




#     pcolor(L0_T1s, phi0s, lens_outer, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     plt.tight_layout()
#     plt.savefig(f'final_extension_outer.pdf',dpi=200)
#     plt.show()






#     pcolor(L0_T1s, phi0s, lens_middle, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)



#     contour(L0_T1s, phi0s, angle_middle, upscale=100, levels=[buckle_thresh,135])




#     plt.tight_layout()
#     plt.savefig(f'final_extension_middle.pdf',dpi=200)
#     plt.show()


#     pcolor(L0_T1s, phi0s, lens_outer_belt, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)



#     contour(L0_T1s, phi0s, angle_outer_belt, upscale=100, levels=[buckle_thresh,135])




#     plt.tight_layout()
#     plt.savefig(f'final_extension_outer_belt.pdf',dpi=200)
#     plt.show()


#     pcolor(L0_T1s, phi0s, lens_middle_no_cable, vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar()
#     plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
#     plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

#     contour(L0_T1s, phi0s, angle_middle_no_cable, upscale=100, levels=[buckle_thresh,135])

#     plt.tight_layout()
#     plt.savefig(f'final_extension_middle_no_cable.pdf',dpi=200)
#     plt.show()


    # pcolor(L0_T1s, phi0s, lens_outer_no_cable, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar()
    # plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
    # plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

    # if np.any(angle_outer_no_cable>buckle_thresh):
    #     contour(L0_T1s, phi0s, angle_outer_no_cable, upscale=100, levels=[buckle_thresh,135])

    # plt.tight_layout()
    # plt.savefig(f'final_extension_outer_no_cable.pdf',dpi=200)
    # plt.show()



    # pcolor(L0_T1s, phi0s, lens_outer_no_remodel, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar()
    # plt.ylabel(r'$\phi_{0}$',fontsize=fontsize)
    # plt.xlabel(r'$L_{0}^{\rm T1}\;(\mu {\rm m})$',fontsize=fontsize)

    # if np.any(angle_outer_no_remodel>buckle_thresh):
    #     contour(L0_T1s, phi0s, angle_outer_no_remodel, upscale=100, levels=[buckle_thresh,135])

    # plt.tight_layout()
    # plt.savefig(f'final_extension_outer_no_remodel.pdf',dpi=200)
    # plt.show()