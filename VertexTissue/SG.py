from .EventExecutor import EventExecutor
from .globals import inner_arc, outer_arc, belt_strength, t_1, t_2, t_belt, t_intercalate, inter_edges, pit_centers, t_pit

def arc_activator(G, arc):
    def f(*args):
        for i in range(0,len(arc)):
            G[arc[i-1]][arc[i]]['myosin'] = belt_strength   
    return f

def pit_activator(G, centers):
    def f(*args):
        myo = 1.5*belt_strength
        for node in centers: 
            for neighbor in G.neighbors(node): 
                G[node][neighbor]['myosin'] = myo

    return f

def arcs_and_pit(G,belt):
    events = [(t_1,  arc_activator(G, inner_arc),"Inner arc established"),
            (t_2,    arc_activator(G, outer_arc),"Outer arc established"),
            (t_belt, arc_activator(G, belt),"Belt established"),
            (t_pit,  pit_activator(G, pit_centers),"Pit activated")
            ]

    return EventExecutor(events)

def just_arcs(G, belt):
    events = [(t_1,  arc_activator(G, inner_arc),"Inner arc established"),
            (t_2,    arc_activator(G, outer_arc),"Outer arc established"),
            (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return EventExecutor(events)

def just_belt(G, belt):
    events = [
            (t_belt, arc_activator(G, belt),"Belt established")
            ]

    return EventExecutor(events)

def arcs_with_intercalation(G, belt):
    inter=False
    def f(t, force_dict):
        nonlocal inter
        # update myosin on inner arc 
        if t == t_1:
            for i in range(0,len(inner_arc)):
                G[inner_arc[i-1]][inner_arc[i]]['myosin'] = belt_strength     
            print("Inner arc established")

        if t >= t_intercalate and not inter:
            inter=True
            for e in inter_edges:
                G[e[0]][e[1]]['myosin'] =  3*belt_strength

        # update myosin on belt
        if t == t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

    return f