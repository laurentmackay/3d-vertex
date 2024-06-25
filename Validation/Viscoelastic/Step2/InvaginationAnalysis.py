import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from VertexTissue.Sweep import sweep
from VertexTissue.Dict import  dict_product, last_dict_key, last_dict_value, take_dicts, dict_mask

from Step2_bis import angle_timeseries, depth_timeline, extension_timeseries, final_arc_ratio, final_cone_slope, final_depth, final_inter_arc_distance, final_width, inter_arc_distance_timeline, intercalations, run, phi0s, phi0_SLS,  base_path, kws_baseline,   final_angle, L0_T1s, final_inter_arc_depth, final_lumen_depth
from Step2_bis import naught_middle_remodel, naught_outer_remodel, naught_double_remodel, extension_timeline

from Step2_bis import kws_strong_pit_middle, kws_strong_pit_double, kws_strong_pit_outer, kws_strong_pit_baseline, kws_middle, kws_double, kws_outer, kws_middle_basal, kws_middle_basal_hi, kws_middle_fine, kws_middle_no_scale, kws_baseline_no_scale, kws_outer_no_scale, kws_middle_smolpit, kws_outer_smolpit, kws_baseline_smolpit
from Step2_bis import clinton_baseline, clinton_double, clinton_outer, clinton_middle, kws_baseline_thresh, kws_baseline_thresh_extend, kws_baseline_thresh_sym, ecs, kws_baseline_thresh_no_scale, kws_baseline_thresh_no_scale_extend, kws_baseline_thresh_no_scale_sym
from Step2_bis import kws_baseline_thresh_no_scale_no_T1, kws_baseline_thresh_no_scale_no_T1_extend, kws_baseline_thresh_no_scale_no_T1_sym, kws_baseline_thresh_no_scale_no_T1_edge, kws_baseline_thresh_no_scale_no_T1_extend_edge, kws_baseline_thresh_no_scale_no_T1_sym_edge, kws_SLS_baseline_thresh,  kws_SLS_baseline_thresh_ext,  kws_SLS_baseline_thresh_con

from VertexTissue.Plotting import pcolor, contour, hatched_contour


kws_contract = kws_SLS_baseline_thresh_con
kws_extend = kws_SLS_baseline_thresh_ext
kws_sym = kws_SLS_baseline_thresh

phi0s=phi0_SLS


##################################################################
###################### INVAGINATION DEPTH ########################
##################################################################

depth_func = final_depth
refresh=False

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
plt.get_current_fig_manager().set_window_title('Depth (Basal)')
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
    # ax.tick_params(labelsize = tick_style['fontsize'])
    ax.set_title(title) 
    plt.xlabel(r'$\varepsilon_c$')

plt.sca(axs[-1])
# pcolor( ecs, 1-phi0s, 50*(depth_sym-(depth_baseline+depth_extend-2*depth_baseline[0,0])-depth_baseline[0,0])/(depth_baseline[0,0]), vmin=0, vmax=vmax)
plt.colorbar(fraction=0.05,pad=.2,)
hatched_contour( ecs, 1-phi0s, (depth_sym-(depth_baseline+depth_extend-2*depth_baseline[0,0])-depth_baseline[0,0])/(depth_baseline[0,0]), upscale=4, levels=[ .05,25], alpha=0.1, hatch_alpha=.5)



# plt.legend(loc='upper left', bbox_to_anchor=(1.05, .975))
# plt.tight_layout()   
# plt.savefig('invagination_depth_VE.pdf')
# plt.savefig('invagination_depth_VE.png', dpi=200)

fig=plt.figure()
fig.set_size_inches(6.5, 3.5)
slice=7
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
# plt.savefig('asymmetric_vs_symmetric_depth_change.png', dpi=200)



depth_timeline_sym = sweep(phi0s, run, kw=kws_sym, pre_process = depth_timeline,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)



depth_timeline_ext = sweep(phi0s, run, kw=kws_extend, pre_process = depth_timeline,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)


depth_timeline_con = sweep(phi0s, run, kw=kws_contract, pre_process = depth_timeline,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=refresh, dtype=float)


### TIMESERIES

plt.figure()
for ts, phi0 in zip(depth_timeline_ext[:, slice], phi0s):
        if ts is not None:
                plt.plot(ts[:,0], ts[:,1],label=f'$\delta = {round(1-phi0,2)}$')
plt.legend()
plt.show()



##################################################################
###################### INVAGINATION WIDTH ########################
##################################################################

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
