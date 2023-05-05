import numpy as np

try:
    from VertexTissue.PyQtViz import edge_view
    import matplotlib.pyplot as plt
    viewable=True
    base_path = './data/'
except:
    viewable=False
    base_path = '/scratch/st-jjfeng-1/lmackay/data/'


from Validation.Viscoelastic.Step5.Step5 import run, kws_baseline_pit, final_depth, sigma

from VertexTissue.Sweep import sweep


depth_func = final_depth

depth_middle  = sweep([1.0,], run, kw=kws_baseline_pit, pre_process = depth_func,
cache=True, savepath_prefix=base_path, inpaint=np.nan, refresh=False)

plt.plot(sigma, np.squeeze(depth_middle), marker='.')

plt.show()