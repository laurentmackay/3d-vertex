import numpy as np

from VertexTissue.Player import pickle_player

def isfinite(x):
    return np.isfinite(x)*1.0

pickle_player(path='./data/viscoelastic/', pattern='shear_8_60_dt_0.1_*.pickle', attr={'tau': isfinite}, speedup=60, cell_edges_only=True, apical_only=True)