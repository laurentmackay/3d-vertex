import os
import pickle
import time

import networkx as nx
import numpy as np
from Tissue import get_triangles, get_outer_belt, new_topology

import globals as const
from funcs import *


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



# calculate volume
vol = convex_hull_volume_bis(np.random.random((6,3)))  

def vertex_integrator(G, G_apical, pre_callback=None, ndim=3, player=False, save_pattern = const.save_pattern, save_rate=1.0, maxwell=False):
    if pre_callback is None or not callable(pre_callback):
        pre_callback = lambda t,f : None

    force_dict = {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 
    blacklist = [] 

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
    


    #@profile
    def integrate(dt, t_final, t=0, save_pattern = save_pattern, player = player, ndim=ndim, save_rate=save_rate, maxwell=maxwell):
        nonlocal G, G_apical, centers,  pre_callback, force_dict

        if ndim == 3:
            compute_forces=compute_tissue_forces_3D
        elif ndim == 2:
            compute_forces=compute_elastic_forces
        
        # num_inter = 0 
        if (player or save_pattern) and len(os.path.split(save_pattern)[0])>1:
            save_path, pattern = os.path.split(save_pattern)
            if len(save_path)>1 and not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path='.'
   

        print(t) 
        pre_callback(t, force_dict)
        t_last_save = t
        file_name = save_pattern.replace('*',str(t))
        with open(file_name, 'wb') as file:
            pickle.dump(G, file)

        if player:

            from Player import pickle_player
            pickle_player(path=save_path, pattern=pattern, start_time=t, attr='myosin', cell_edges_only=True, apical_only=True, parallel=True)

        t0 = time.time()
        
        while t <= t_final:

            


            force_dict = {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 
            pre_callback(t, force_dict)
            compute_forces()
            
            for node in force_dict:
                G.node[node]['pos'][:ndim] += (dt/const.eta)*force_dict[node]  #forward euler step for nodes


            if maxwell:
                for e in G.edges():
                    a, b = e[0], e[1]
                    pos_a = pos[a]
                    pos_b = pos[b]
                    
                    tau = G[a][b]['tau']
                    r= dt/tau
                    if r:
                        dist = euclidean_distance(pos_a,pos_b)
                        G[a][b]['l_rest'] += dt*(dist - G[a][b]['l_rest'])/tau

            check_for_intercalations(t)

            # increment t by dt
            t = t + dt
            t1=time.time()
            print(dt, t,f'{t1-t0} seconds elapsed') 
            t0 = t1
            
            if abs((t - t_last_save) - save_rate)  <= dt/2 : 
                t_last_save = t
                file_name = save_pattern.replace('*',str(t))
                with open(file_name, 'wb') as file:
                    pickle.dump(G, file)
    
    def compute_elastic_forces():
        nonlocal force_dict, pos

        pos = nx.get_node_attributes(G,'pos')
        

        l_rest = nx.get_edge_attributes(G,'l_rest')
        myosin = nx.get_edge_attributes(G,'myosin')
        
        for i, e in enumerate(G.edges()):
            
            a, b = e[0], e[1]
            pos_a = pos[a]
            pos_b = pos[b]
            direction, dist = unit_vector_and_dist(pos_a,pos_b)

            
            # dists[i] = dist
            # drx[i] = direction
            magnitude = mu_apical*(dist - l_rest[e])
            magnitude2 = myo_beta*myosin[e]
            force = (magnitude + magnitude2)*direction[:ndim]

            force_dict[a] += force
            force_dict[b] -= force

    def compute_tissue_forces_3D():
        nonlocal force_dict, circum_sorted, triangles, pos

        compute_elastic_forces()
        
        # pre-calculate magnitude of pressure
        # index of list corresponds to index of centers list
        PI = np.zeros(len(centers),dtype=float) 
        # eventually move to classes?
        for n in range(len(centers)):
            # get nodes for volume
            pts = get_points(G,centers[n],pos) 
            # calculate volume
            vol = convex_hull_volume_bis(pts)  
            # calculate pressure
            PI[n] = -press_alpha*(vol-const.v_0) 


        
        for center, pts, pressure in zip(centers, circum_sorted, PI):  
            for i in range(len(pts)):
                for inds in ((center,pts[i],pts[i-1]),(center+basal_offset,pts[i-1]+basal_offset,pts[i]+basal_offset)):
                    pos_face =np.array([pos[j] for j in inds])
                    _, area_vec, _, _ = be_area_2(pos_face,pos_face)                       

                    force = pressure*area_vec/3.0
                    force_dict[inds[0]] += force
                    force_dict[inds[1]] += force
                    force_dict[inds[2]] += force
    


        # pressure for side panels
        # loop through each cell
        for cell_nodes, pressure in zip(circum_sorted, PI):
            # loop through the faces
            for i in range(len(cell_nodes)):
                pts_id = (cell_nodes[i-1], cell_nodes[i], cell_nodes[i]+basal_offset, cell_nodes[i-1]+basal_offset)
                pts_pos = np.array([pos[pts_id[ii]] for ii in range(4)])
                # on each face, calculate the center
                center = np.average(pts_pos,axis=0)
                # loop through the 4 triangles that make the face
                for ii in range(0,4):
                    pos_side = np.array([center, pts_pos[ii-1], pts_pos[ii]])
                    _, area_vec = area_side(pos_side) 
                    
                    force = pressure*area_vec/2.0
                    force_dict[pts_id[ii-1]] += force
                    force_dict[pts_id[ii]] += force
        
        # Implement bending energy
        # Loop through all alpha, beta pairs of triangles
        offset=0
        for pair in triangles:
            for offset in (0, basal_offset):
                alpha = [i+offset for i in pair[0]]
                beta = [i+offset for i in pair[1]]
                
                # Apical faces, calculate areas and cross-products z
                pos_alpha = np.array([pos[i] for i in alpha])
                pos_beta = np.array([pos[i] for i in beta])
                A_alpha, A_alpha_vec, A_beta, A_beta_vec = be_area_2(pos_alpha, pos_beta)

                for inda, node in enumerate(alpha):
                    # inda = alpha.index(node) 
                    nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
                    if node in beta:
                        indb = beta.index(node)
                        nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 

                        frce = const.c_ab * bending_energy_2(True, True,A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[-1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[-1]])
                    else:
                        frce = const.c_ab * bending_energy_2(True, False, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]], pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]])
                    
                    force_dict[node] += frce

                for indb, node in enumerate(beta):
                    # don't double count the shared nodes
                    nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                    if node not in alpha:
                        # frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
                        frce = const.c_ab*bending_energy_2(False, True, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_beta[0]], pos[nbhrs_beta[1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[1]])

                        force_dict[node] += frce

    def check_for_intercalations(t):
        nonlocal circum_sorted, triangles, G_apical, blacklist

        pos = nx.get_node_attributes(G,'pos')
        node=0
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
                                print("Intercalation event between nodes", node, "and", neighbor, "at t = ", t) 
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

                                # sever connections
                                # apical   
                                G.remove_edge(node,cents[0])
                                G.remove_edge(node,temp1[0])
                                G.remove_edge(neighbor,cents[1])
                                G.remove_edge(neighbor,temp2[0])
                                # basal 
                                G.remove_edge(node+basal_offset,cents[0]+basal_offset)
                                G.remove_edge(node+basal_offset,temp1[0]+basal_offset)
                                G.remove_edge(neighbor+basal_offset,cents[1]+basal_offset)
                                G.remove_edge(neighbor+basal_offset,temp2[0]+basal_offset)

                                # add new connections
                                # apical 
                                # new edges 
                                G.add_edge(node,temp2[0],l_rest = const.l_apical, myosin=0,color='#808080')
                                G.add_edge(neighbor,temp1[0],l_rest = const.l_apical, myosin=0,color='#808080')
                                # new spokes 
                                G.add_edge(neighbor,ii,l_rest = const.l_apical, myosin=0)
                                G.add_edge(node,jj,l_rest = const.l_apical, myosin=0)
                                # basal 
                                # new edges 
                                G.add_edge(node+basal_offset,temp2[0]+basal_offset,l_rest = const.l_apical, myosin=0,color='#808080')
                                G.add_edge(neighbor+basal_offset,temp1[0]+basal_offset,l_rest = const.l_apical, myosin=0,color='#808080')
                                # new spokes 
                                G.add_edge(neighbor+basal_offset,ii+basal_offset,l_rest = const.l_apical, myosin=0)
                                G.add_edge(node+basal_offset,jj+basal_offset,l_rest = const.l_apical, myosin=0)
                                
                                # reset myosin on contracted edge
                                G[node][neighbor]['myosin'] = 0
                                G[node+basal_offset][neighbor+basal_offset]['myosin'] = 0
                                
                                blacklist.append([min(node, neighbor), max(node, neighbor)])
                                
                                circum_sorted, triangles, G_apical = new_topology(G_apical, [node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
                                G.graph['circum_sorted'] = circum_sorted
                                node-=1
                                break
                    j += 1

            node +=1
        return
    return integrate

