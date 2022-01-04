from numpy.core.numeric import Infinity
from globals import inner_arc, outer_arc, belt_strength, t_1, t_2, t_belt, t_intercalate, inter_edges
import numpy as np
import networkx as nx

from globals import basal_offset

def invagination(G, belt, centers):
    rate=0.01
    def f(t, t_prev=-Infinity):
        # update myosin on inner arc 
        if t >= t_1 and t_prev<t_1:
            for i in range(0,len(inner_arc)):
                G[inner_arc[i-1]][inner_arc[i]]['myosin'] = belt_strength     
            print("Inner arc established")

        # update myosin on outer arc 
        if t >= t_2 and t_prev<t_2:
            for i in range(0,len(outer_arc)):
                G[outer_arc[i-1]][outer_arc[i]]['myosin'] = belt_strength     
            print("Outer arc established")

        # update myosin on belt
        if t >= t_belt and t_prev<t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

        if t >= t_intercalate and t_prev<t_intercalate:
            for e in inter_edges:
                G[e[0]][e[1]]['myosin'] =  3*belt_strength

        r=np.random.rand()
        if t_prev>0 and r/(t-t_prev)<rate:
            myo = nx.get_edge_attributes(G,'myosin')
            
            candidates = [key for (key, val) in  myo.items() if val==0 and key[0]<1000 and key[1]<1000 and key[0] not in centers and key[1] not in centers
             and np.all(np.array([e['myosin'] for e in G[key[0]].values()])==0) and np.all(np.array([e['myosin'] for e in G[key[1]].values()])==0)]
            if len(candidates):
                e = candidates[np.random.randint(len(candidates))]
                
                strength=3*belt_strength
                
                G[e[0]][e[1]]['myosin']=strength
                G[e[0]+basal_offset][e[1]+basal_offset]['myosin']=strength

                print(f'intercalation shecduled for {e}')
            
    return f
