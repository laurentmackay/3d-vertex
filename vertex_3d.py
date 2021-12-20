import time
import warnings

import networkx as nx
import numpy as np

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

def vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=None):
    if pre_callback is None or not callable(pre_callback):
        pre_callback = lambda t : None

    force_dict={}
    
    l_rest = nx.get_edge_attributes(G,'l_rest')
    dists = np.zeros((len(l_rest),))
    drx = np.zeros((len(l_rest),3))
                    
                    

    #@profile
    def integrate(dt,t_final, t=0):
        nonlocal G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback, force_dict, dists, l_rest
        
        num_inter = 0 
        blacklist = [] 
        contract = [True for _ in range(num_inter)]

        print(t) 
        file_name = 't_fast' + str(int(t)) 
        nx.write_gpickle(G, file_name + '.pickle')
        np.save(file_name, circum_sorted) 
        t0 = time.time()
        
        pre_callback(t)
        compute_forces()
        force_dict_prev=force_dict
        non_zero = len(np.argwhere([np.any(np.abs(f)>10**-12) for f in force_dict.values()]))
        if non_zero:
            warnings.warn(f'{non_zero} vertices have non-zero intial forces, these forces will have no effect on the vertex movement')


        while t <= t_final:

            
            # increment t by dt
            # initialize force_dict back to zeros
            t = round(t+dt,1)
            t1=time.time()
            print(dt, t,f'{t1-t0} seconds elapsed') 
            t0=t1

            pre_callback(t)

            compute_forces()

            #### UPDATE STATES ####
            for i, e in enumerate(G.edges()):
                drx[i]=unit_vector(G.node[e[0]]['pos'],G.node[e[1]]['pos'])

            keq=320
            for node in force_dict:
                G.node[node]['pos'] = G.node[node]['pos'] + (force_dict[node]-force_dict_prev[node])/keq  #forward euler step for nodes

            r=dt/const.tau

            for i, e in enumerate(G.edges()):
                strain = (dists[i]/l_rest[e])-1.0
                if np.abs(strain)>0.1:
                    delta = (dists[i]-l_rest[e])*r
                    G[e[0]][e[1]]['l_rest'] = l_rest[e] + delta
                    
                        # G[e[0]][e[1]]['l_rest'] = l_rest[e] + delta
                    G.node[e[0]]['pos'] = G.node[e[0]]['pos'] + delta*drx[i]
                    G.node[e[1]]['pos'] = G.node[e[1]]['pos'] - delta*drx[i]




            force_dict_prev=force_dict

        ## Check for intercalation events
            pos = nx.get_node_attributes(G,'pos')
            for node in range(0,num_api_nodes):
                if node not in belt: 
                    nhbrs = list(G.neighbors(node))
                    for neighbor in nhbrs:
                        if (neighbor < basal_offset) and (neighbor not in belt) and (node not in centers) and (neighbor not in centers) and ([min(node, neighbor), max(node, neighbor)] not in blacklist): 
                        
                            a = pos[node]
                            b = pos[neighbor]

                            
                            dist = euclidean_distance(a,b)
                            
                            if (dist < const.l_intercalation): 
                                c = pos[node+basal_offset]
                                d = pos[neighbor+basal_offset]

                                print("Intercalation event between nodes", node, "and", neighbor, "at t = ", t) 
                                # collapse nodes to same position 
                                # apical  
                                avg_loc = (a + b) / 2.0 
                                a = avg_loc 
                                b = avg_loc 
                                # basal 
                                avg_loc = (c + d) / 2.0 
                                c = avg_loc 
                                d = avg_loc 
                                # move nodes toward new center 
                                # apical 
                                cents = list(set(G.neighbors(node)) & set(G.neighbors(neighbor)))
                                mvmt = unit_vector(a,pos[cents[1]])
                                G.node[node]['pos'] = a + l_mvmt*mvmt
                                mvmt = unit_vector(b,pos[cents[0]])
                                G.node[neighbor]['pos'] = b + l_mvmt*mvmt
                                # basal 
                                #cents = list(set(G.neighbors(node+basal_offset)) & set(G.neighbors(neighbor+basal_offset)))
                                mvmt = unit_vector(c,pos[cents[1]+basal_offset])
                                G.node[node+basal_offset]['pos'] = c + l_mvmt*mvmt
                                mvmt = unit_vector(d,pos[cents[0]+basal_offset])
                                G.node[neighbor+basal_offset]['pos'] = d + l_mvmt*mvmt

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
                                
                                circum_sorted, triangles, K = new_topology(K,[node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
                                
                                # if min(node,neighbor) == 301:
                                #     contract[0] = False

        #    #set dt for next loop 
        #    if var_dt == True:
        #        if any(contract) == True:
        #            # if any edges are still contracting, check for threshold length 
        #            for i in range(0,num_inter):
        #            # calculate lengths of those that are still True 
        #                if contract[i] == True:
        #                    a = inter_edges[i][0]
        #                    b = inter_edges[i][1]
        #                    if euclidean_distance(pos[a],pos[b]) < 0.2:
        #                        dt = 0.1
        #                        break 
        #        else: 
        #            if isclose(t % 1, 0) == False:       
        #                dt = 0.1 
        #            else:
        #                dt = const.dt
        #                var_dt = False 
        #    else:
        #        dt  = const.dt

        # Save nx Graph in pickled form for plotting later
            
            if t % 1 == 0: 
                file_name = 't_fast' + str(round(t)) 
                nx.write_gpickle(G,file_name + '.pickle')
                np.save(file_name,circum_sorted)

    def compute_forces():
        nonlocal G, centers, circum_sorted, triangles, pre_callback, force_dict, dists, l_rest, drx

        pos = nx.get_node_attributes(G,'pos')
        force_dict = {new_list: np.zeros(3,dtype=float) for new_list in G.nodes()} 
        
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

        l_rest = nx.get_edge_attributes(G,'l_rest')
        myosin = nx.get_edge_attributes(G,'myosin')
        
        for i, e in enumerate(G.edges()):
            
            a, b = e[0], e[1]
            pos_a = pos[a]
            pos_b = pos[b]
            direction, dist = unit_vector_and_dist(pos_a,pos_b)

            
            dists[i] = dist
            drx[i] = direction
            magnitude = mu_apical*(dist - l_rest[e])
            magnitude2 = myo_beta*myosin[e]
            force = (magnitude + magnitude2)*direction

            force_dict[a] += force
            force_dict[b] -= force
        
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
                    
                    direction = area_vec 
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

    return integrate


