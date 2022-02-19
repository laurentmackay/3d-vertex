from VertexTissue.Player import pickle_player


pickle_player(path='./data/viscoelastic/', pattern='shear_2_60_dt_0.001_*.pickle', attr='myosin', speedup=15, cell_edges_only=True, apical_only=True)