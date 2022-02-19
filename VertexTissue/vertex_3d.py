from dis import dis
import os
import pickle
import time

import networkx as nx
import numpy as np


from .Tissue import Tissue_Forces, get_triangles, get_outer_belt, new_topology
from . import globals as const
from .funcs import *
from .Player import pickle_player
from . PyQtViz import edge_viewer
from VertexTissue import Tissue


# dimensions of the cell 
l_apical = const.l_apical 
l_depth = const.l_depth 

# Set the arcs list
inner_arc = const.inner_arc
outer_arc = const.outer_arc

# mechanical parameters
# rest lengths of the passively elastic apical, basal and cell walls
l0_apical = l_apical
l0_basal = l_apical 
l0_wall = l_depth 

mu_apical = const.mu_apical         
mu_basal = const.mu_basal          
mu_wall = const.mu_wall          
myo_beta = const.myo_beta 
eta = const.eta 
press_alpha = const.press_alpha 
l_mvmt = const.l_mvmt
basal_offset=const.basal_offset

z3=np.zeros((3,))
cross33(z3,z3)

unit_vector(z3,z3)
unit_vector_and_dist(z3,z3)
euclidean_distance(z3,z3)
bending_energy_2(True, True,z3, 1.0 , z3, 1.0, z3, z3, z3, z3)
be_area_2(np.tile(z3,reps=(3,1)),np.tile(z3,reps=(3,1)))
area_side(np.tile(z3,reps=(3,1)))
vol = convex_hull_volume_bis(np.random.random((6,3)))  

def vertex_integrator(G, G_apical, pre_callback=None, post_callback=None, ndim=3, player=False, viewer=False,  save_pattern = const.save_pattern, save_rate=1.0, maxwell=False, adaptive=False, length_prec=0.05, minimal=False):
    if pre_callback is None or not callable(pre_callback):
        pre_callback = lambda t, f : None

    if post_callback is None or not callable(post_callback):
        post_callback = lambda t, f : None

    force_dict = {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 
    dists = {node: 0 for node in G.edges()} 
    drx ={node: np.zeros(ndim ,dtype=float) for node in G.edges()} 
    blacklist = [] 

    view = None

    pos = nx.get_node_attributes(G,'pos')
    pos_apical = nx.get_node_attributes(G_apical ,'pos')

    if 'circum_sorted' in G.graph.keys():
        circum_sorted = G.graph['circum_sorted']
    else:
        circum_sorted = None

    if 'centers' in G.graph.keys():
        centers = G.graph['centers']
    else:
        centers = None

    belt = get_outer_belt(G_apical)
    triangles = get_triangles(G_apical, pos_apical, centers, belt)

    if 'num_apical_nodes' in G.graph.keys():
        num_api_nodes=G.graph['num_apical_nodes']
    else:
        num_api_nodes=0
    
    # @jit(nopython=True, cache=True)


    compute_forces = Tissue_Forces(G, ndim=ndim, minimal=minimal)

    def init_force_dict():
        return {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 

    #@profile
    def integrate(dt, t_final, dt_init=None, dt_min = None, t=0, save_pattern = save_pattern, player = player, ndim=ndim, save_rate=save_rate, maxwell=maxwell, strain_thresh=0, length_prec=length_prec, adaptive=adaptive, adaptation_rate=0.1, **kw):
        nonlocal G, G_apical, centers,  pre_callback, force_dict, drx, dists, view
        if dt_init is None:
            h=dt
        else:
            h=dt_init

        if adaptive and dt_min is None:
            dt_min = 0
        
        def timestep_bound():

            dot = np.array([dists[e]/np.sum((force_dict[e[0]]-force_dict[e[1]])*drx[e]) for e in G.edges()])
            return np.maximum(np.minimum(length_prec*np.nanmin(np.abs(dot))*const.eta, dt), dt_min)


        
        # num_inter = 0 
        if (player or save_pattern is not None) and len(os.path.split(save_pattern)[0])>1:
            save_path, pattern = os.path.split(save_pattern)
            if len(save_path)>1 and not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path='.'
   

        print(t) 
        pre_callback(t, force_dict)
        if save_pattern:
            t_last_save = t
            file_name = save_pattern.replace('*',str(t))
            with open(file_name, 'wb') as file:
                pickle.dump(G, file)

        if player:
            pickle_player(path=save_path, pattern=pattern, start_time=t, attr='myosin', cell_edges_only=True, apical_only=True, parallel=True)

        if viewer:
            view = edge_viewer(G,attr='myosin', cell_edges_only=True, apical_only=True)

        t0 = time.time()

        force_dict = init_force_dict() 



        pre_callback(t, force_dict)

        dists, drx  = compute_forces.compute_distances_and_directions()
        compute_forces(force_dict=force_dict, compute_distances=False, **kw)

        post_callback(t, force_dict)

        while t <= t_final:

            
            for node in force_dict:
                G.node[node]['pos'][:ndim] += (h/const.eta)*force_dict[node]  #forward euler step for nodes

            old_dists=dists
            # dists = {node: 0 for node in G.edges()} 
            # drx ={node: np.zeros(ndim ,dtype=float) for node in G.edges()} 
            dists, drx = compute_forces.compute_distances_and_directions()

            if maxwell:
                for e in G.edges():
                    a, b = e[0], e[1]
                    pos_a = pos[a]
                    pos_b = pos[b]
                    
                    tau = G[a][b]['tau']
                    r = h/(tau*2)
                    if r:
                        dist = dists[e]
                        l_rest = G[a][b]['l_rest']
                        eps = (dist - l_rest)
                        if abs(eps)/l_rest>strain_thresh:
                            G[a][b]['l_rest'] = (l_rest*(1-r)+r*(dist+old_dists[e]))/(1+r) #Crank-Nicholson
                            # G[a][b]['l_rest'] += h*eps/tau #Forward-Euler

            check_for_intercalations(t)

            # increment t by dt
            t = t + h
            t1=time.time()
            # print(h, t,f'{t1-t0} seconds elapsed') 
            t0 = t1

            force_dict = init_force_dict()
            pre_callback(t, force_dict)
            
            compute_forces(force_dict=force_dict, compute_distances=False, **kw)
            
            post_callback(t, force_dict)

            if adaptive:
                hnew=timestep_bound()
                if hnew>h:
                    h=(1-adaptation_rate)*h+adaptation_rate*hnew
                else:
                    h=hnew
            else:
                h=dt

            # print(h, t)
            if save_pattern and abs((t - t_last_save) - save_rate)  <= h/2 : 
                t_last_save = t
                file_name = save_pattern.replace('*',str(t))
                with open(file_name, 'wb') as file:
                    pickle.dump(G, file)

            if viewer:
                view(G, title = f't={t}')

    def check_for_intercalations(t):
        nonlocal circum_sorted, triangles, G_apical, blacklist, view, dists, drx

        pos = nx.get_node_attributes(G,'pos')
        node=0
        intercal = False
        while node<num_api_nodes:
        # for node in range(0,num_api_nodes):
            if node not in belt: 
                nhbrs=list(G.neighbors(node))
                j=0
                while j<len(nhbrs):
                    neighbor=nhbrs[j]
                    if (neighbor < basal_offset) and (neighbor not in belt) and (node not in centers) and (neighbor not in centers) and ([min(node, neighbor), max(node, neighbor)] not in blacklist): 
                    
                        a = pos[node]
                        b = pos[neighbor]
                        c = pos[node+basal_offset]
                        d = pos[neighbor+basal_offset]
                        
                        dist = euclidean_distance(a,b)
                        
                        if (dist < const.l_intercalation): 
                            if (np.random.rand(1)[0] < 1.):
                                intercal=True
                                print("Intercalation event between nodes", node, "and", neighbor, "at t = ", t) 


                                if viewer:
                                    view(G, title = f'about to intercalate; t={t}')
                                # collapse nodes to same position 
                                # apical  
                                avg_loc = (np.array(a) + np.array(b)) / 2.0 
                                a = avg_loc 
                                b = avg_loc 
                                # basal 
                                avg_loc = (np.array(c) + np.array(d)) / 2.0 
                                c = avg_loc 
                                d = avg_loc 
                                # move nodes toward new center 
                                # apical 
                                cents = list(set(G.neighbors(node)) & set(G.neighbors(neighbor)))
                                mvmt = unit_vector(a,pos[cents[1]])
                                a = np.array([a[0]+l_mvmt*mvmt[0], a[1]+l_mvmt*mvmt[1], a[2]+l_mvmt*mvmt[2]])
                                G.node[node]['pos'] = a 
                                mvmt = unit_vector(b,pos[cents[0]])
                                b = np.array([b[0]+l_mvmt*mvmt[0], b[1]+l_mvmt*mvmt[1], b[2]+l_mvmt*mvmt[2]])
                                G.node[neighbor]['pos'] = b 
                                # basal 
                                #cents = list(set(G.neighbors(node+basal_offset)) & set(G.neighbors(neighbor+basal_offset)))
                                mvmt = unit_vector(c,pos[cents[1]+basal_offset])
                                c = np.array([c[0]+l_mvmt*mvmt[0], c[1]+l_mvmt*mvmt[1], c[2]+l_mvmt*mvmt[2]])
                                G.node[node+basal_offset]['pos'] = c 
                                mvmt = unit_vector(d,pos[cents[0]+basal_offset])
                                d = np.array([d[0]+l_mvmt*mvmt[0], d[1]+l_mvmt*mvmt[1], d[2]+l_mvmt*mvmt[2]])
                                G.node[neighbor+basal_offset]['pos'] = d 
                                
                                ii = list((set(list(G.neighbors(node))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))[0]
                                jj = list((set(list(G.neighbors(neighbor))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))[0]
                                temp1 = list(set(G.neighbors(node)) & set(G.neighbors(cents[0])))
                                temp1.remove(neighbor)
                                temp2 = list(set(G.neighbors(neighbor)) & set(G.neighbors(cents[1])))
                                temp2.remove(node)


                                for offset in (0, basal_offset):

                                    old_rods = np.array(((node, cents[0]), (node, temp1[0]),(neighbor, cents[1]),(neighbor, temp2[0])))+offset
                                    new_rods = np.array(((node, temp2[0]), (node, jj),(neighbor, temp1[0]),(neighbor, ii)))+offset
                                    
                                    old_attrs = [ G[s[0]][s[1]] for  s in old_rods]
                                                
                                    # # sever connections
                                    for spoke in old_rods:
                                        G.remove_edge(*spoke)
                                    
                                    # # add new connections
                                    # new edges 
                                    G.add_edge(*new_rods[0],**old_attrs[3])
                                    G.add_edge(*new_rods[2],**old_attrs[1])
                                    # new spokes 
                                    G.add_edge(*new_rods[3], **G[node+offset][ii+offset])
                                    G.add_edge(*new_rods[1], **G[neighbor+offset][jj+offset])

                                    # # reset myosin on contracted edge
                                    G[node+offset][neighbor+offset]['myosin'] = 0
                                
                                blacklist.append([min(node, neighbor), max(node, neighbor)])
                                
                                circum_sorted, triangles, G_apical = new_topology(G_apical, [node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
                                G.graph['circum_sorted'] = circum_sorted
                                G.graph['triangles'] = triangles
                                node-=1
                                dists, drx = compute_forces.compute_distances_and_directions()
                                if viewer:
                                    view(G, title = f'freshly resolved intercalation t={t}')
                                    print('resolved')
                                break
                    j += 1

            node +=1
        return intercal
    return integrate

