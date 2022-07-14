from VertexTissue.Tissue import get_outer_belt
from .Events import EventExecutor
from .globals import inner_arc, outer_arc, belt_strength, pit_strength, t_1, t_2, t_belt, t_intercalate, inter_edges, pit_centers, t_pit

def arc_activator(G, arc, strength=belt_strength ):
    def f(*args):
        if arc[0] in G[arc[-1]].keys(): #complete the loop, if the arc is a loop
            G[arc[-1]][arc[0]]['myosin'] = strength 

        for i in range(1,len(arc)):
            if arc[i] in G[arc[i-1]].keys():
                G[arc[i-1]][arc[i]]['myosin'] = strength   
            else:
                print(f'egde {arc[i-1]}:{arc[i]} not present in graph')
    return f

def pit_activator(G, centers, strength=pit_strength):
    def f(*args):
        for node in centers: 
            for neighbor in G.neighbors(node):
                if neighbor in G[node].keys():
                    G[node][neighbor]['myosin'] = strength

    return f

def intercalation_activator(G, edges, strength=1000, basal=False):
    def f(*args):
        if basal:
            basal_offset=G.graph['basal_offset']
        for e in edges:
            if e[1] in G[e[0]].keys():
                G[e[0]][e[1]]['myosin'] =  strength
                if basal:
                    G[e[0]+basal_offset][e[1]+basal_offset]['myosin'] =  strength

    return f

def arcs_and_pit(G,arcs=(inner_arc,outer_arc), t_arcs=t_1, t_pit=t_pit, arc_strength=belt_strength):
    events = [*[(t_arcs, arc_activator(G, arc, strength=arc_strength),f'Arc #{i+1} established') for i, arc in enumerate(arcs)],
            (t_pit,  pit_activator(G, pit_centers),"Pit activated")
            ]

    return EventExecutor(events)

def arc_pit_and_intercalation (G, belt, arcs=(inner_arc,outer_arc), inter_edges=inter_edges, basal_intercalation=False, intercalation_strength=1000, arc_strength=belt_strength, belt_strength=belt_strength, pit_strength=pit_strength, t_belt=t_belt, t_intercalate=t_intercalate):

    events = [*[(t_1, arc_activator(G, arc, strength=arc_strength),f'Arc #{i+1} established') for i,arc in enumerate(arcs)],
            (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            (t_belt, arc_activator(G, belt, strength=belt_strength),"Belt established"),
            (t_pit,  pit_activator(G, pit_centers, strength=pit_strength),"Pit activated")
            ]

    return EventExecutor(events)

def just_arcs(G, belt):
    events = [(t_1,  arc_activator(G, inner_arc),"Inner arc established"),
            (t_2,    arc_activator(G, outer_arc),"Outer arc established"),
            (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return EventExecutor(events)

def just_belt(G, belt, t=t_belt, strength=belt_strength):
    events = [(t, arc_activator(G, belt, strength=strength),"Belt established"),]

    return EventExecutor(events)

def just_pit(G):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
            ]

    return EventExecutor(events)

def pit_and_belt(G,belt):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
              (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return EventExecutor(events)

def pit_and_intercalation(G, basal_intercalation=False, intercalation_strength=1000):
    events = [(t_pit,  pit_activator(G, pit_centers),"Pit activated"),
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return EventExecutor(events)

def belt_and_intercalation(G, belt, inter_edges=inter_edges, basal_intercalation=False, intercalation_strength=1000, t_belt=t_belt, t_intercalate=t_intercalate, belt_strength=belt_strength):
    events = [(t_belt,  arc_activator(G, belt, strength=belt_strength),"Belt activated"),
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return EventExecutor(events)

def arcs_and_intercalation(G, arcs, inter_edges=inter_edges, basal_intercalation=False, intercalation_strength=1000, t_belt=t_belt, t_intercalate=t_intercalate, arc_strength=belt_strength):
    events = [*[(t_belt,  arc_activator(G, arc, strength=arc_strength),f"Arc #{i+1} activated") for i, arc in enumerate(arcs)],
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return EventExecutor(events)

def just_intercalation(G,inter_edges, basal_intercalation=False, intercalation_strength=1000, t_intercalate=t_intercalate):
    events = [
              (t_intercalate,    intercalation_activator(G, inter_edges, basal=basal_intercalation, strength=intercalation_strength),"Intercalations triggered"),
            ]

    return EventExecutor(events)
    
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
            for e in inter_edges:
                G[e[0]][e[1]]['myosin'] =  1000
                if basal:
                    G[e[0]+basal_offset][e[1]+basal_offset]['myosin'] =  3*belt_strength

        # update myosin on belt
        if t == t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

    return f


# def just_intercalation(G, belt, basal=False):
#     inter=False
    
#     def f(t, force_dict):
#         nonlocal inter

#         if t >= t_intercalate and not inter:
#             inter=True
#             for e in inter_edges:
#                 G[e[0]][e[1]]['myosin'] =  3*belt_strength
#                 if basal:
#                     G[e[0]+basal_offset][e[1]+basal_offset]['myosin'] =  3*belt_strength



#     return f


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