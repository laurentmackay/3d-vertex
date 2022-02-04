from VertexTissue.Player import pickle_player


pickle_player(path='./data/testing/', pattern='viscoelastic_tau_{tau}_*.pickle',attr='myosin', cell_edges_only=True, apical_only=True)