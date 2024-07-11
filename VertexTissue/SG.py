'''Events/EventListeners for the Drosophila Salivary Gland (SG) model. To be used with the `ResearchTools.Events` module.'''

from collections.abc import Iterable

import numpy as np

from ResearchTools.Events import TimeBasedEventExecutor
from VertexTissue.globals import inner_arc, outer_arc, belt_strength, pit_strength, t_1, t_2, t_belt, t_intercalate, inter_edges_middle, pit_centers, t_pit, intercalation_strength

def arc_activator(G, arc, strength=belt_strength ):
    '''Returns myosin activation function which for the whole `arc` of edges in the network `G`.'''
    def activate_arc(arc):
        if len(arc)<2:
            return
            
        if arc[0] in G[arc[-1]].keys(): #complete the loop, if the arc is a loop
            G[arc[-1]][arc[0]]['myosin'] = strength 

        for i in range(1,len(arc)):
            if arc[i] in G[arc[i-1]].keys():
                G[arc[i-1]][arc[i]]['myosin'] = strength   
            else:
                print(f'egde {arc[i-1]}:{arc[i]} not present in graph')

    def f(*args):
        if not isinstance(arc[0], Iterable):
            activate_arc(arc)
        else:
            for sub_arc in arc:
                activate_arc(sub_arc)
 
    return f

def pit_activator(G, centers, strength=pit_strength, edge_ratio=0):
    '''Returns a spoke myosin activation function which acts on cells with centers `centers` in the network `G`,
    this function takes a single argument which is a 2-tuple specifying the edge to activate.
    
    If `edge_ratio` is non-zero, then edges have myosin strength given by `strength*edge_ratio`.
    '''
    zippable=isinstance(strength,Iterable) and len(strength)==len(centers)

    def f(*args):
        if edge_ratio:
            cirucm_sorted = G.graph['circum_sorted']
        for node in centers: 
            for neighbor in G.neighbors(node):
                if neighbor in G[node].keys():
                    G[node][neighbor]['myosin'] = strength

            if edge_ratio:
                c=np.argwhere(G.graph['centers']==node)[0,0]
                i = cirucm_sorted[c][0]
                for j in [*cirucm_sorted[c][1:], i]:
                    G[i][j]['myosin'] = strength*edge_ratio
                    i=j

                
    def fzip(*args):
        if edge_ratio:
            cirucm_sorted = G.graph['circum_sorted']

        for node, s in zip(centers,strength): 
            for neighbor in G.neighbors(node):
                if neighbor in G[node].keys():
                    G[node][neighbor]['myosin'] = s

            if edge_ratio:
                c=np.argwhere(G.graph['centers']==node)[0,0]
                i = cirucm_sorted[c][0]
                for j in [*cirucm_sorted[c][1:], i]:
                    G[i][j]['myosin'] = s*edge_ratio
                    i=j
    
    if zippable:
        return fzip
    else:
        return f

def edge_activator(G, strength=intercalation_strength, basal=False):
    '''Returns a base myosin activation function which acts on the network `G`,
    this function takes a single argument which is a 2-tuple specifying the edge to activate.'''
    if basal:
        basal_offset=G.graph['basal_offset']

    def activate(edge):
        if edge is None:
            return 
            
        if edge[1] in G[edge[0]].keys():
            G[edge[0]][edge[1]]['myosin'] =  strength

            if basal:
                G[edge[0]+basal_offset][edge[1]+basal_offset]['myosin'] =  strength

    return activate

def intercalation_activator(G, edges, strength=intercalation_strength, basal=False):
    '''Returns an event function for the activation of intercalation events for all `edges` in the network `G`.'''
    activate = edge_activator(G, strength=strength, basal=basal)

    if edges and np.isscalar(edges[0]):
        edges=(edges,)

    def f(*args):

        for e in edges:
            activate(e)

    return f

def arcs_and_pit(G,arcs=(inner_arc,outer_arc), t_arcs=t_1, t_pit=t_pit, arc_strength=belt_strength):
    '''Returns a `ResearchTools.TimeBasedEventExecutor` loaded with event functions for arc and pit contraction.'''
    events = [*[(t_arcs, arc_activator(G, arc, strength=arc_strength),f'Arc #{i+1} established') for i, arc in enumerate(arcs)],
            (t_pit,  pit_activator(G, pit_centers),"Pit activated")
            ]

    return TimeBasedEventExecutor(events)

def arcs_pit_and_intercalation (G, belt, arcs=(inner_arc,outer_arc), inter_edges=inter_edges_middle, basal_intercalation=False, intercalation_strength=1000, arc_strength=belt_strength, belt_strength=belt_strength, pit_strength=pit_strength, t_belt=t_belt, t_intercalate=t_intercalate,t_1=t_1, pit_centers=pit_centers, edge_ratio=0):
    '''Returns a `ResearchTools.TimeBasedEventExecutor` loaded with event functions for arc and pit contraction as well as any intercalations.'''
    
    events = [*[(t_1, arc_activator(G, arc, strength=arc_strength),f'Arc #{i+1} established') for i,arc in enumerate(arcs)],
            (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            (t_belt, arc_activator(G, belt, strength=belt_strength),"Belt established"),
            (t_pit,  pit_activator(G, pit_centers, strength=pit_strength, edge_ratio=edge_ratio),"Pit activated")
            ]

    return TimeBasedEventExecutor(events)

def just_arcs(G, belt):
    events = [(t_1,  arc_activator(G, inner_arc),"Inner arc established"),
            (t_2,    arc_activator(G, outer_arc),"Outer arc established"),
            (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return TimeBasedEventExecutor(events)

def just_belt(G, belt, t=t_belt, strength=belt_strength):
    events = [(t, arc_activator(G, belt, strength=strength),"Belt established"),]

    return TimeBasedEventExecutor(events)

def just_pit(G):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
            ]

    return TimeBasedEventExecutor(events)

def pit_and_belt(G,belt):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
              (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return TimeBasedEventExecutor(events)

def pit_and_intercalation(G, basal_intercalation=False, intercalation_strength=1000):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
              (t_intercalate,    intercalation_activator(G, inter_edges_middle, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return TimeBasedEventExecutor(events)

def belt_and_intercalation(G, belt, inter_edges=inter_edges_middle, basal_intercalation=False, intercalation_strength=1000, t_belt=t_belt, t_intercalate=t_intercalate, belt_strength=belt_strength):
    events = [(t_belt,  arc_activator(G, belt, strength=belt_strength),"Belt activated"),
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return TimeBasedEventExecutor(events)

def arcs_and_intercalation(G, arcs, inter_edges=inter_edges_middle, basal_intercalation=False, intercalation_strength=1000, t_belt=t_belt, t_intercalate=t_intercalate, arc_strength=belt_strength):
    events = [*[(t_belt,  arc_activator(G, arc, strength=arc_strength),f"Arc #{i+1} activated") for i, arc in enumerate(arcs)],
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return TimeBasedEventExecutor(events)

def just_intercalation(G,inter_edges, basal_intercalation=False, intercalation_strength=1000, t_intercalate=t_intercalate):
    events = [
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return TimeBasedEventExecutor(events)
    
# inter_edges=((39,40),(45,46),(277,276),(272,274))
def arcs_with_intercalation(G, belt, basal=False):
    inter=False
    def f(t, force_dict, basal=False):
        nonlocal inter
        # update myosin on inner arc 
        if t == t_1:
            for i in range(0,len(inner_arc)):
                G[inner_arc[i-1]][inner_arc[i]]['myosin'] = belt_strength     
            print("Inner arc established")

        if t >= t_intercalate and not inter:
            inter=True
            for e in inter_edges_middle:
                G[e[0]][e[1]]['myosin'] =  1000
                if basal:
                    G[e[0]+basal_offset][e[1]+basal_offset]['myosin'] =  3*belt_strength

        # update myosin on belt
        if t == t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

    return f



def convergent_extension_test(G):
    inter_edges=((146,157),  (94, 95), (295, 284), (85,73), (35,34), (246, 245), (109, 98), (272, 273), (13, 6), (100,113), (298, 311), (276, 287), (40,51))
    inter=False
    def f(t, force_dict):
        nonlocal inter

        if t >= t_intercalate and not inter:
            inter=True
            for e in inter_edges:
                G[e[0]][e[1]]['myosin'] =  3*belt_strength

    return f