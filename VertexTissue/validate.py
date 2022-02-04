import os

import networkx as nx
import numpy as np







def validate(n,attr='pos'): 
    try:
        G=nx.read_gpickle(f't{n}.pickle')
        G1=nx.read_gpickle(f't_fast{n}.pickle')

        vals = nx.get_node_attributes(G,attr)
        vals1 = nx.get_node_attributes(G1,attr)

        max_diff  = np.max([np.max(np.abs(v-v1)) for v,v1 in zip(vals.values(),vals1.values()) ])
        print(f'{n}: {max_diff}')


        return True
    except:
        return False


for i in range(300):
    cont= validate(i)
    if not cont:
        break