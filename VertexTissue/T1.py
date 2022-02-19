from .EventExecutor import EventExecutor
from .globals import inner_arc, outer_arc, belt_strength, t_1, t_2, t_belt, t_intercalate, inter_edges, pit_centers, t_pit



def simple_T1(G,e=(11,1), strength=belt_strength, basal=False):
    def activate(force_dict):
        G[e[0]][e[1]]['myosin']=strength
        if basal:
            G[e[0]+1000][e[1]+1000]['myosin']=strength

    return EventExecutor([(0,activate)])