import time

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

def vertex_integrator(G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback=None, intercalation_callback=None, length_prec=0.05):
    if pre_callback is None or not callable(pre_callback):
        pre_callback = lambda t : None
    if intercalation_callback is None or not callable(intercalation_callback):
        intercalation_callback = lambda node, neighbor : None

    force_dict = {}
    l_rest={}
    N_edges = len(G.edges())
    drx=np.zeros((N_edges,3))
    dists=np.zeros((N_edges,))
    blacklist = [] 
    #@profile
    def integrate(dt,t_final, t=0):
        nonlocal G, K, centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback, force_dict, l_rest, dists, drx
        
        num_inter = 0 
        
        contract = [True for counter in range(0,num_inter)]

        print(t) 
        pre_callback(t)

        file_name = 't_fast' + str(int(t)) 
        nx.write_gpickle(G, file_name + '.pickle')
        np.save(file_name, circum_sorted) 
        t0 = time.time()


        compute_forces()
        delta_F = np.array([ force_dict[e[0]]-force_dict[e[1]] for e in G.edges])
        dtmax=-length_prec*dists/np.sum(drx*delta_F/const.eta,axis=1)
        dt_curr=np.minimum(np.min(dtmax[np.argwhere(dtmax>0)]),dt)



        while t <= t_final:

            
            # increment t by dt
            # t = round(t+dt_curr,1)


            pre_callback(t+dt_curr, t_prev=t)
            
            t += dt_curr
            t1=time.time()
            print(dt_curr, t,f'{t1-t0} seconds elapsed') 
            t0=t1
            compute_forces()

            delta_F = np.array([ force_dict[e[0]]-force_dict[e[1]] for e in G.edges])
            dtmax=-length_prec*dists/np.sum(drx*delta_F/const.eta,axis=1)
            dt_curr=np.minimum(np.min(dtmax[np.argwhere(dtmax>0)]),dt)
            # print(dt_curr)

            for node in force_dict:
                G.node[node]['pos'] = G.node[node]['pos'] + (dt_curr/const.eta)*force_dict[node]  #forward euler step for nodes


            
        
            

            r=dt_curr/const.tau
            for i, e in enumerate(G.edges()):
                dist=dists[i]
                strain = (dist/l_rest[e])-1.0
                if np.abs(strain)>0.01:
                    G[e[0]][e[1]]['l_rest'] = (l_rest[e]+dist*r)/(1.0+r)

            

            check_for_intercalations(t)
                            

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
        nonlocal force_dict, circum_sorted, triangles, l_rest, dists, drx

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

    def check_for_intercalations(t):
        nonlocal circum_sorted, triangles, K, blacklist

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

                        dist = euclidean_distance(a,b)
                        
                        if (dist < const.l_intercalation): 

                            c = pos[node+basal_offset]
                            d = pos[neighbor+basal_offset]
                        
                            print("Intercalation event between nodes", node, "and", neighbor, "at t = ", t) 
                            # collapse nodes to same position 
                            # apical  
                            a0=a
                            b0=b
                            avg_loc = (a + b) / 2.0 
                            a = avg_loc 
                            b = avg_loc 
                            # basal 
                            avg_loc = (c + d) / 2.0 
                            c0=c
                            d0=d
                            c = avg_loc 
                            d = avg_loc 
                            # move nodes toward new center 
                            # apical 
                            cents = list(set(G.neighbors(node)) & set(G.neighbors(neighbor)))
                            mvmt = unit_vector(a,pos[cents[1]])
                            a = a + l_mvmt*mvmt
                            G.node[node]['pos'] = a 
                            mvmt = unit_vector(b,pos[cents[0]])
                            b = b + l_mvmt*mvmt
                            G.node[neighbor]['pos'] = b 
                            # basal 
                            #cents = list(set(G.neighbors(node+basal_offset)) & set(G.neighbors(neighbor+basal_offset)))
                            mvmt = unit_vector(c,pos[cents[1]+basal_offset])
                            c = c + l_mvmt*mvmt
                            G.node[node+basal_offset]['pos'] = c 
                            mvmt = unit_vector(d,pos[cents[0]+basal_offset])
                            d = d + l_mvmt*mvmt
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
                            
                            circum_sorted, triangles, K = new_topology(K,[node, neighbor], cents, temp1, temp2, ii, jj, belt, centers, num_api_nodes)
                            G.graph['circum_sorted']=circum_sorted

                            intercalation_callback(node,neighbor)
                            node-=1
                            break
                    j += 1
            node += 1



    return integrate



###############
def tissue_3d():

    def gen_nodes(ori,z):
        nodes = [[ori[0] + r*np.cos(n*np.pi/3), ori[1] + r*np.sin(n*np.pi/3),z] for n in range(0,6)]
        return np.array(nodes)

    def add_nodes(nodes, i):
        pos = nx.get_node_attributes(G,'pos')
        cen_index = i-1
        if i < 1000:
            centers.append(cen_index)
        AS_boundary = []
        spokes = []
        for node in nodes:
            add_node = True
            for existing_node in pos:
                if euclidean_distance(pos[existing_node],node) < 10**(-7):
                    add_node = False
                    AS_boundary.append(existing_node)
                    spokes.append((cen_index,existing_node))
                    break

            if add_node == True:
                G.add_node(i,pos=node)
                i += 1
                AS_boundary.append(i-1)
                spokes.append((cen_index,i-1))

        return AS_boundary, spokes, i

    def add_spokes_edges(spokes, boundary):
        boundary.append(boundary[0])
        G.add_edges_from(spokes, l_rest=const.l_apical, myosin=0)
        attr = {'l_rest' : const.l_apical, 'myosin':0, 'color':'#808080'}
        if nx.__version__>"2.3":
            nx.classes.function.add_path(G,boundary, **attr)
        else:
            G.add_path(boundary, **attr)

        return

    G = nx.Graph()
    if nx.__version__>"2.3":
        G.node=G._node

    r = const.l_apical              # initial spoke length
    num_cells = 2*const.hex-1          # number of cells in center row

    centers = []
    
    # Apical Nodes
    i = 0
    # Center cell set up
    z = 0.0
    origin = [0.0,0.0,z]
    G.add_node(i,pos=np.array(origin))
    i += 1

    nodes = gen_nodes(origin,z)
    AS_boundary, spokes, i = add_nodes(nodes,i)
    add_spokes_edges(spokes, AS_boundary)

    for index in range(1,int((num_cells - 1)/2.)+1):
        # # Step Up
        origin = [0, np.sqrt(3)*r*index,0.0]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

        # # # Step down
        origin = [0, -np.sqrt(3)*r*index,0.0]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

    for index in range(1,const.hex):  
        if (num_cells - index) % 2 == 0:
            for j in range(1,(num_cells-index),2):
                origin = [(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

            # Step Left

                origin = [-(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [-(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

        else:
            for j in range(0,(num_cells-index),2):
                origin = [3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

                # Step Left
                origin = [-3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [-3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)
   
    circum_sorted = []
    pos = nx.get_node_attributes(G,'pos') 
    xy = np.array([[pos[n][0],pos[n][1]] for n in range(0,i)])
    for center in centers:
        a, b = sort_corners(list(G.neighbors(center)),xy[center],xy)
        circum_sorted.append(np.asarray([b[n][0] for n in range(len(b))]))
    circum_sorted = np.array(circum_sorted)

    G.graph['circum_sorted']=circum_sorted

    belt = []
    for node in G.nodes():
        if len(list(G.neighbors(node))) < 6:
            belt.append(node)
    xy_belt = [xy[n] for n in belt]
    a,b = sort_corners(belt, xy[0], xy)
    belt = np.array([b[n][0] for n in range(len(b))])
    
    # make swaps for the "corner" nodes that the angle doesn't account for
    for n in range(1,len(belt)-1):
        if G.has_edge(belt[n-1],belt[n]) == False:
            belt[n], belt[n+1] = belt[n+1], belt[n]

    triangles = []
    for node in G.nodes():
        if node not in belt:
            if node in centers:
                out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
                alpha_beta = [[[node,neighbors[k-1],neighbors[k-2]],[node, neighbors[k],neighbors[k-1]]] for k in range(0,6)]

                for entry in alpha_beta:
                    triangles.append(entry)
            else: # node not a center, so that I don't double count pairs, only keep those that cross a cell edge
                out1, out2 = sort_corners(list(G.neighbors(node)),pos[node],pos)
                neighbors = [out2[k][0] for k in range(0,len(out2))]
	            
                for k in range(0,6):
                    alpha = [node,neighbors[k-1],neighbors[k-2]]
                    beta = [node,neighbors[k],neighbors[k-1]]
                    
                    if set(alpha) & set(centers) != set(beta) & set(centers):
                        triangles.append([alpha,beta])

    print("Apical nodes added correctly.")
    print("Number of apical nodes are", i)
    
    G2D = G.copy()
    if nx.__version__>"2.3":
        G2D.node = G2D._node

    num_apical_nodes = i
    
    # Basal Nodes
    i = 1000
    z = -const.l_depth
    # Center cell set up
    origin = [0.0,0.0,z]
    G.add_node(i,pos=np.array(origin))
    i += 1

    nodes = gen_nodes(origin,z)
    AS_boundary, spokes, i = add_nodes(nodes,i)
    add_spokes_edges(spokes, AS_boundary)

    for index in range(1,int((num_cells - 1)/2.)+1):
        # # Step Up
        origin = [0, np.sqrt(3)*r*index,z]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

        # # # Step down
        origin = [0, -np.sqrt(3)*r*index,z]
        G.add_node(i,pos=np.array(origin))
        i += 1

        nodes = gen_nodes(origin,z)
        AS_boundary, spokes, i = add_nodes(nodes,i)
        add_spokes_edges(spokes, AS_boundary)

    for index in range(1,const.hex):  
        if (num_cells - index) % 2 == 0:
            for j in range(1,(num_cells-index),2):
                origin = [(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

            # Step Left

                origin = [-(3/2.)*r*index,(np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

                origin = [-(3/2.)*r*index,(-np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)

        else:
            for j in range(0,(num_cells-index),2):
                origin = [3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

                # Step Left
                origin = [-3*(1/2.)*r*index, (np.sqrt(3)/2.)*r*j,z]
                G.add_node(i,pos=np.array(origin))
                i += 1

                nodes = gen_nodes(origin,z)
                AS_boundary, spokes, i = add_nodes(nodes,i)
                add_spokes_edges(spokes, AS_boundary)
                
                if j != 0:
                    origin = [-3*(1/2.)*r*index, -(np.sqrt(3)/2.)*r*j,z]
                    G.add_node(i,pos=np.array(origin))
                    i += 1

                    nodes = gen_nodes(origin,z)
                    AS_boundary, spokes, i = add_nodes(nodes,i)
                    add_spokes_edges(spokes, AS_boundary)

    print("Basal Nodes Added")
    for n in range(0,num_apical_nodes):
        if n not in centers:
            G.add_edge(n,n+basal_offset, l_rest = const.l_depth, myosin = 0, beta = 0)

    print("Lateral Connections made")
    
    return G, G2D, centers, num_apical_nodes, circum_sorted, belt, triangles