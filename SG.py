from globals import inner_arc, outer_arc, belt_strength, t_1, t_2, t_belt, t_intercalate, inter_edges


def just_arcs(G, belt):

    def f(t):
        # update myosin on inner arc 
        if t == t_1:
            for i in range(0,len(inner_arc)):
                G[inner_arc[i-1]][inner_arc[i]]['myosin'] = belt_strength     
            print("Inner arc established")

        # update myosin on outer arc 
        if t == t_2:
            for i in range(0,len(outer_arc)):
                G[outer_arc[i-1]][outer_arc[i]]['myosin'] = belt_strength     
            print("Outer arc established")

        # update myosin on belt
        if t == t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

    return f

def arcs_with_intercalation(G, belt):

    def f(t):
        # update myosin on inner arc 
        if t == t_1:
            for i in range(0,len(inner_arc)):
                G[inner_arc[i-1]][inner_arc[i]]['myosin'] = belt_strength     
            print("Inner arc established")

        if t == t_intercalate :
            for e in inter_edges:
                G[e[0]][e[1]]['myosin'] =  3*belt_strength

        # update myosin on belt
        if t == t_belt:
            for i in range(0,len(belt)):
                G[belt[i-1]][belt[i]]['myosin'] = belt_strength     
            print("Belt established") 

    return f