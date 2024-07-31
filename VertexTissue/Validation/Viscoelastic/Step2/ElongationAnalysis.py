import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from VertexTissue.Tissue import tissue_3d
from VertexTissue.globals import inter_edges_middle, inter_edges_middle_bis, inter_edges_outer, inter_edges_outer_bis, inner_arc, outer_arc, pit_strength, myo_beta, l_apical, press_alpha

from ResearchTools.Geometry import euclidean_distance, unit_vector, unit_vector_and_dist
from ResearchTools.Iterable import imin, imax
from ResearchTools.Util import find
from ResearchTools.Sweep import sweep


from VertexTissue.Validation.Viscoelastic.Step2.Step2_bis import final_elongation_ratio, average_elongation_ratio, average_apparent_elongation_ratio, final_apparent_elongation_ratio
from VertexTissue.Validation.Viscoelastic.Step2.Step2_bis import angle_timeseries, depth_timeline, extension_timeseries, final_arc_ratio, final_cone_slope, final_depth, final_inter_arc_distance, final_width, inter_arc_distance_timeline, intercalations, run, phi0s, phi0_SLS,  base_path, kws_baseline,   final_angle, L0_T1s, final_inter_arc_depth, final_lumen_depth
from VertexTissue.Validation.Viscoelastic.Step2.Step2_bis import extension_timeline,  ecs, kws_SLS_baseline_thresh,  kws_SLS_baseline_thresh_ext,  kws_SLS_baseline_thresh_con

from VertexTissue.Plotting import pcolor, contour, hatched_contour, add_colorbar_to_side, spectral_rainbow, make_lines_rainbow


kws_contract = kws_SLS_baseline_thresh_con
kws_extend = kws_SLS_baseline_thresh_ext
kws_sym = kws_SLS_baseline_thresh

phi0s=phi0_SLS

G, G_apical = tissue_3d( hex=7,  basal=True)

refresh=False
ppkw={'vertical_projection':True}


elongation_ratio_baseline  = sweep(phi0s, run, kw=kws_contract, pre_process = final_apparent_elongation_ratio, pre_process_kw=ppkw,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

elongation_ratio_extend  = sweep(phi0s, run, kw=kws_extend, pre_process = final_apparent_elongation_ratio, pre_process_kw=ppkw,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)

elongation_ratio_sym  = sweep(phi0s, run, kw=kws_sym, pre_process = final_apparent_elongation_ratio, pre_process_kw=ppkw,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)


##################################################################
################## ELONGATION RATIO HEATMAP ######################
##################################################################

fig, axs = plt.subplots(1,3)
fig.set_size_inches(9.5, 3.5)
# plt.get_current_fig_manager().canvas.set_window_title('Middle')
axs=axs.ravel()
# for i in range(mid.shape[-1]):
        # plt.sca(axs[i])
plt.get_current_fig_manager().set_window_title('Depth (Basal)')

cmap = mpl.colors.Colormap('viridis')
ratios = (elongation_ratio_baseline, elongation_ratio_extend, elongation_ratio_sym)
vmax = np.log2(max(*[np.nanmax(d) for d in ratios ]))
plt.sca(axs[0])
plt.ylabel('$\delta$')
for r, ax, title in zip(ratios, axs,
                    ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
    plt.sca(ax)
    pcolor( ecs, 1-phi0s, np.log2(r), vmin=0, vmax=vmax)
    # ax.tick_params(labelsize = tick_style['fontsize'])
    ax.set_title(title,usetex=True, fontsize=16) 
    plt.xlabel(r'$\varepsilon_c$')
    # contour( ecs, 1-phi0s, r, upscale=4, levels=[ 2,2.5], color='w')
    # ax.clabel(CS, inline=True, fontsize=10)
plt.tight_layout()
# pcolor( ecs, 1-phi0s, 50*(depth_sym-(depth_baseline+depth_extend-2*depth_baseline[0,0])-depth_baseline[0,0])/(depth_baseline[0,0]), vmin=0, vmax=vmax)





for r, ax, title in zip(ratios, axs,
                    ('Asymmetric VE: Contraction','Asymmetric VE: Extension', 'Symmetric VE')):
    plt.sca(ax)
    cb_ax=add_colorbar_to_side(ax, side='bottom')
    cb_ax.yaxis.label.set_size(16)
    yticks = list(range(4))
    ylbls = [ plt.Text(1,p,f'$\\mathdefault{{{2**p}}}$') for p in yticks]

    cb_ax.set_xticks(yticks, labels=ylbls)

    plt.sca(ax)
    CS=contour( ecs, 1-phi0s, r, upscale=4, sampling='linear', levels=[ 2,2.5], color='w')
    ax.clabel(CS, inline=True, fontsize=10)





fig.savefig('elongation_ratio_VE.pdf', bbox_extra_artists=[cb_ax], bbox_inches='tight')
plt.show()