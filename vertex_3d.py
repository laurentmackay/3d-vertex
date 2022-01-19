import os
import time
import pickle


import networkx as nx
import numpy as np

from numba import jit, prange
from numba.typed import List

import globals as const
from funcs import *
from tissue_3d import topological_mesh




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


    l_rest={}
    N_edges = len(G.edges())
    drx=np.zeros((N_edges,3))
    dists=np.zeros((N_edges,1))
    dists_old=np.zeros((N_edges,1))
    # PI = np.zeros(len(centers),dtype=float)

    ind_dict = {n:i for i, n in enumerate(G.nodes()) }
    neighbors = {n: tuple(map( lambda i : ind_dict[i],G.neighbors(n))) for n in G.nodes()}
    edges=np.array([np.array((ind_dict[e[0]],ind_dict[e[1]])) for e in G.edges()])

    myosin = np.fromiter(nx.get_edge_attributes(G,'myosin').values(),dtype=float).reshape((-1,1))
    l_rest = np.fromiter(nx.get_edge_attributes(G,'l_rest').values(),dtype=float).reshape((-1,1))

    forces_prev=None

    @jit(nopython=True, cache=True)
    def timestep_bound(forces,drx,edges):
        delta_F = forces[edges[:,0]]-forces[edges[:,1]]
        dtmax=length_prec*dists/np.sum(drx*delta_F/const.eta,axis=1).reshape((-1,1))
        return np.min(np.abs(dtmax))

    # @profile
    def integrate(dt,t_final, t=0, save_rate=None, save_pattern=const.save_pattern, maxwell=True, visco=False):

        if save_pattern and len(os.path.split(save_pattern)[0])>1:
            save_path = os.path.split(save_pattern)[0]
            if len(save_path)>1 and not os.path.exists(save_path):
                os.makedirs(save_path)

        dt=float(dt)
        nonlocal G, K, edges,  centers, num_api_nodes, circum_sorted, belt, triangles, pre_callback, l_rest, dists, drx, ind_dict, myosin, l_rest, forces_prev
        
        num_inter = 0 
        
        contract = [True for counter in range(0,num_inter)]

        if save_rate is not None:
            with open(save_pattern.replace('*',str(t)),'wb') as file: 
                pickle.dump(G, file)
            last_save=t
        t0 = time.time()

        pos = np.array([*nx.get_node_attributes(G,'pos').values()])
 

        if maxwell:
            update=update_pos_maxwell
        else:
            update=update_pos

        print(t) 
        pre_callback(t)
        dists, drx = compute_edge_distance_and_direction(pos, edges, dists, drx)
        vols = compute_cell_volumes(pos)
        forces = compute_forces2(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)
        # force_dict_prev = force_dict

  
        h=np.min((timestep_bound(forces,drx,edges),dt))

        # pre_callback(t+h)
        # vols = compute_cell_volumes(pos)
        # dists, drx = compute_edge_distance_and_direction(pos, edges, dists, drx)
        # forces=compute_forces(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)
        # force_dict_prev = force_dict

        # h=np.minimum(timestep_bound(forces,drx,edges),dt)


        # dt_curr = dt

        if maxwell:
            forces_prev=forces
        
        while t <= t_final:

            pos = np.array([*nx.get_node_attributes(G,'pos').values()])

            
            

            myosin = np.fromiter(nx.get_edge_attributes(G,'myosin').values(),dtype=float).reshape((-1,1))
            l_rest = np.fromiter(nx.get_edge_attributes(G,'l_rest').values(),dtype=float).reshape((-1,1))
            vols = compute_cell_volumes(pos)
            dists_old[:]=dists[:]
            # force_dict_prev_prev = force_dict_prev
            # force_dict_prev = force_dict

            h=update(h, dt, pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)

            for i,node in enumerate(G.nodes()):
                G.node[node]['pos'] = pos[i]   #forward euler step for nodes


            
            
            dists, drx = compute_edge_distance_and_direction(pos, edges, dists, drx)

            # r=dt_curr/(2.0*const.tau)
            if not maxwell and visco:
                r=h/(const.tau)
                for i, e in enumerate(G.edges()):
                    dist_old = dists_old[i]
                    dist=dists[i]
                    strain = (dist/l_rest[i])-1.0
                    # if np.abs(strain)>0.01:# and (l_rest[i]<l_apical or (dist<l_apical and dist_old<l_apical)):
                    G[e[0]][e[1]]['l_rest'] = (l_rest[i]*(1.0-r) + r*(dist_old +dist))/(1+r)
                    # G[e[0]][e[1]]['l_rest']   += r*(dist_old-l_rest[i])
                # G[e[0]][e[1]]['l_rest'] = (l_rest[e]+r*dist)/(1.0+r)

            

            check_for_intercalations(t)

            pre_callback(t+h, t_prev=t)
            t += h
            t1=time.time()
            print(f'{t1-t0} seconds elapsed') 
            t0=t1
                            

        # Save nx Graph in pickled form for plotting later
            
            if save_rate is not None and t-last_save>=save_rate:
                with open(save_pattern.replace('*',str(t)),'wb') as file: 
                    pickle.dump(G, file)
                last_save=t

    two_thirds=2/3
    three_quarters=3/4
    # @jit(nopython=True, cache=True)
    def  update_pos(h, dt, pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers):
        
        forces=compute_forces2(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)


        h1=timestep_bound(forces,drx,edges)
        h=min((h1,dt,h))

        


        k1=forces/const.eta

        
        pos2 = pos + h*two_thirds*k1

        dists, drx = compute_edge_distance_and_direction(pos2, edges, dists, drx)
        forces = compute_forces2(pos2, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)


        h2=timestep_bound(forces,drx,edges)
        h=min((h1,h2,dt))
        if h2<h1 and h2<dt:
            pos2 = pos + h*two_thirds*k1
            dists, drx = compute_edge_distance_and_direction(pos2, edges, dists, drx)
            forces = compute_forces2(pos2, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)



        k2=forces/const.eta

        

        pos += h*(k1/4+three_quarters*k2)

        return h


    def  update_pos_maxwell(h, dt, pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers):
        nonlocal forces_prev
        forces=compute_forces2(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)


        


        
        pos += (forces-forces_prev)/1000+dt*(forces/const.eta)

        forces_prev=forces

        return dt

    @jit(nopython=True, cache=True)
    def compute_edge_distance_and_direction(pos, edges, dists, drx):
        for i in range(len(edges)):
            e=edges[i]
            a, b = e[0], e[1]
            pos_a = pos[a]
            pos_b = pos[b]
            direction, dist = unit_vector_and_dist(pos_a,pos_b)
            dists[i]=dist
            drx[i]=direction
        return dists, drx

    def compute_cell_volumes(pos):
        nonlocal centers, ind_dict
        #alot of this data should just be computed once and stored in G, ideally
        vols=np.zeros((centers.shape[0],))
        for i, center in enumerate(centers):
            # get nodes for volume
            pts=tuple()
            for offset in (0, basal_offset):
                q=center+offset
                pts+=(ind_dict[q],)+neighbors[q]

            pos_pts = pos[pts,:]

            # calculate volume
            vols[i] = convex_hull_volume_bis(pos_pts)  
        
        return vols

    @jit(nopython=True, cache=True)
    def compute_pressure(vols):
        return -press_alpha*(vols-const.v_0)

    # @profile
    # @jit(nopython=True, cache=True)
    # def compute_forces(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers):
    #     PI=-press_alpha*(vols-const.v_0)

    #     num_api_nodes = int(pos.shape[0]/2)

    #     forces = np.zeros(pos.shape)

    #     mag_elastic = mu_apical*(dists - l_rest)
    #     mag_myo = myo_beta*myosin
    #     force_edge = (mag_elastic+ mag_myo)*drx

    #     for e, f in zip(edges, force_edge):
    #         a, b = e[0], e[1]
    #         forces[a] += f
    #         forces[b] -= f
        
    #     #loop through each cell to handle pressure forces
    #     for center, pts, pressure in zip(centers, circum_sorted, PI):
    #         # pressure for apical and basal faces
    #         for j in range(len(pts)):
    #             for inds in (np.array((center,pts[j],pts[j-1])),np.array((center+num_api_nodes,pts[j-1]+num_api_nodes,pts[j]+num_api_nodes))):
    #                 pos_face = pos[inds,:]
    #                 _, area_vec, _, _ = be_area_2(pos_face,pos_face)                       

    #                 force = pressure*area_vec/3.0
                    
    #                 forces[inds] += force


    #         # pressure for side panels
    #         for j in range(len(pts)):
    #             pts_id = np.array((pts[j-1], pts[j], pts[j]+num_api_nodes, pts[j-1]+num_api_nodes))
    #             pts_pos = pos[pts_id,:]
    #             # on each face, calculate the center
    #             center = np.sum(pts_pos,axis=0)/4.0
    #             # loop through the 4 triangles that make the face
    #             for ii in range(0,4):
    #                 pos_side = np.vstack((center, pts_pos[ii-1,:], pts_pos[ii,:]))
    #                 _, area_vec = area_side(pos_side) 
                    
    #                 direction = area_vec 
    #                 force = pressure*area_vec/2.0
    #                 forces[pts_id[ii-1]] += force
    #                 forces[pts_id[ii]] += force
    
                
        
    #     # Implement bending energy
    #     # Loop through all alpha, beta pairs of triangles
    #     for pair in triangles:
    #         for offset in (0, num_api_nodes):
    #             alpha = pair[0]+offset
    #             beta = pair[1]+offset
                
    #             # Apical faces, calculate areas and cross-products z
    #             pos_alpha = pos[alpha]
    #             pos_beta = pos[beta]
    #             A_alpha, A_alpha_vec, A_beta, A_beta_vec = be_area_2(pos_alpha, pos_beta)

    #             for inda, node in enumerate(alpha):
    #                 # inda = alpha.index(node) 
    #                 nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
    #                 # beta_list = beta.tolist()
    #                 if node in beta:
    #                 #     indb = beta_list.index(node)
    #                     indb=np.argwhere(beta==node)[0,0]
    #                     nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 

    #                     frce = const.c_ab * bending_energy_2(True, True,A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[-1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[-1]])
    #                 else:
    #                     frce = const.c_ab * bending_energy_2(True, False, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]], pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]])
                    
    #                 forces[node] += frce

    #             for indb, node in enumerate(beta):
    #                 # don't double count the shared nodes
    #                 nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
    #                 if node not in alpha:
    #                     # frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
    #                     frce = const.c_ab*bending_energy_2(False, True, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_beta[0]], pos[nbhrs_beta[1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[1]])

    #                     forces[node] += frce
    #     return forces


    @jit(nopython=True, cache=True, nogil=True)
    def compute_forces2(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers):
        PI=-press_alpha*(vols-const.v_0)

        num_api_nodes = int(pos.shape[0]/2)

        forces = np.zeros(pos.shape)

        # mag_elastic = mu_apical*(dists - l_rest)
        # mag_myo = myo_beta*myosin
        # force_edge = (mag_elastic+ mag_myo)*drx

        for i in range(len(edges)):
            e=edges[i]
            a, b = e[0], e[1]
            mag_elastic = mu_apical*(dists[i] - l_rest[i])
            mag_myo = myo_beta*myosin[i]
            f = (mag_elastic+ mag_myo)*drx[i]
            forces[a] += f
            forces[b] -= f
        
        #loop through each cell to handle pressure forces
        for i in range(len(PI)):
            pressure=PI[i]
            pts=circum_sorted[i]
            center=centers[i]
            
            for j in range(len(pts)):
                # pressure for apical and basal faces
                for inds in (np.array((center,pts[j],pts[j-1])),np.array((center+num_api_nodes,pts[j-1]+num_api_nodes,pts[j]+num_api_nodes))):
                    pos_face = pos[inds,:]
                    _, area_vec, _, _ = be_area_2(pos_face,pos_face)                       

                    force = pressure*area_vec/3.0
                    
                    forces[inds[0]] += force
                    forces[inds[1]] += force
                    forces[inds[2]] += force


                # pressure for side panels
                pts_id = np.array((pts[j-1], pts[j], pts[j]+num_api_nodes, pts[j-1]+num_api_nodes))
                pts_pos = pos[pts_id,:]
                # on each face, calculate the center
                centroid = np.sum(pts_pos,axis=0)/4.0
                # loop through the 4 triangles that make the face
                for k in range(0,4):
                    pos_side = np.vstack((centroid, pts_pos[k-1,:], pts_pos[k,:]))
                    _, area_vec = area_side(pos_side) 
                    
                    direction = area_vec 
                    force = pressure*area_vec/2.0
                    forces[pts_id[k-1]] += force
                    forces[pts_id[k]] += force
    
                
        
        # Implement bending energy
        # Loop through all alpha, beta pairs of triangles
        for i in range(len(triangles)):
            pair=triangles[i]
            for offset in (0, num_api_nodes):
                alpha = pair[0]+offset
                beta = pair[1]+offset
                
                # Apical faces, calculate areas and cross-products z
                pos_alpha = pos[alpha]
                pos_beta = pos[beta]
                A_alpha, A_alpha_vec, A_beta, A_beta_vec = be_area_2(pos_alpha, pos_beta)

                for inda in range(3):
                    node=alpha[inda]
                    # inda = alpha.index(node) 
                    nbhrs_alpha = (alpha[(inda+1)%3], alpha[(inda-1)%3]) 
                    # beta_list = beta.tolist()
                    if node in beta:
                    #     indb = beta_list.index(node)
                        indb=np.argwhere(beta==node)[0,0]
                        nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 

                        frce = const.c_ab * bending_energy_2(True, True,A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[-1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[-1]])
                    else:
                        frce = const.c_ab * bending_energy_2(True, False, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]], pos[nbhrs_alpha[0]], pos[nbhrs_alpha[1]])
                    
                    forces[node] += frce

                for indb in range(3):
                    node=beta[indb]
                    # don't double count the shared nodes
                    nbhrs_beta = (beta[(indb+1)%3], beta[(indb-1)%3]) 
                    if node not in alpha:
                        # frce = const.c_ab*bending_energy(False, nbhrs_beta, A_alpha, A_beta, pos)
                        frce = const.c_ab*bending_energy_2(False, True, A_alpha_vec, A_alpha , A_beta_vec, A_beta, pos[nbhrs_beta[0]], pos[nbhrs_beta[1]], pos[nbhrs_beta[0]], pos[nbhrs_beta[1]])

                        forces[node] += frce
        return forces


    def check_for_intercalations(t):
        nonlocal circum_sorted, triangles, K
        blacklist = [] 
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
                        
                        a = G.node[node]['pos']
                        b = G.node[neighbor]['pos']

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
                            mvmt = unit_vector(pos[cents[1]],pos[cents[0]])
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
                            remove = ((node,cents[0]),(node,temp1[0]),(neighbor,cents[1]),(neighbor,temp2[0])) 
                            old_edges = []
                            for e in remove:
                                old_edges.append(G[e[0]][e[1]])
                                G.remove_edge(*e)

                            remove_basal = ((node,cents[0]),(node,temp1[0]),(neighbor,cents[1]),(neighbor,temp2[0])) 
                            old_edges_basal = []
                            for e in remove_basal:
                                e=(e[0]+basal_offset, e[1]+basal_offset)
                                old_edges_basal.append(G[e[0]][e[1]])
                                G.remove_edge(*e)
                            
                            # basal 
                            # G.remove_edge(node+basal_offset,cents[0]+basal_offset)
                            # G.remove_edge(node+basal_offset,temp1[0]+basal_offset)
                            # G.remove_edge(neighbor+basal_offset,cents[1]+basal_offset)
                            # G.remove_edge(neighbor+basal_offset,temp2[0]+basal_offset)

                            new_edge = {'myosin':0, 'l_rest':l_apical}
                            new_ab_link_edge = {'myosin':0, 'l_rest':l_depth}

                            # add new connections
                            # apical 
                            # new edges 
                            G.add_edge(node,temp2[0],**old_edges[3])
                            G.add_edge(neighbor,temp1[0],**old_edges[1])
                            # new spokes to new neighbors
                            G.add_edge(neighbor,ii,**G[node][ii])
                            G.add_edge(node,jj,**G[neighbor][jj])
                            # # new edges 
                            # G.add_edge(node,temp2[0],**new_edge)
                            # G.add_edge(neighbor,temp1[0],**new_edge)
                            # # new spokes to new neighbors
                            # G.add_edge(neighbor,ii,**new_edge)
                            # G.add_edge(node,jj,**new_edge)

                            # # basal 
                            # new edges 
                            G.add_edge(node+basal_offset,temp2[0]+basal_offset,**old_edges_basal[3])
                            G.add_edge(neighbor+basal_offset,temp1[0]+basal_offset,**old_edges_basal[1])
                            # new spokes 
                            G.add_edge(neighbor+basal_offset,ii+basal_offset,**G[node+basal_offset][ii+basal_offset])
                            G.add_edge(node+basal_offset,jj+basal_offset,**G[neighbor+basal_offset][jj+basal_offset])

                            # # new edges 
                            # G.add_edge(node+basal_offset,temp2[0]+basal_offset,**new_edge)
                            # G.add_edge(neighbor+basal_offset,temp1[0]+basal_offset,**new_edge)
                            # # new spokes 
                            # G.add_edge(neighbor+basal_offset,ii+basal_offset,**new_edge)
                            # G.add_edge(node+basal_offset,jj+basal_offset,**new_edge)
                            
                            
                            # reset myosin on contracted edge
                            G[node][neighbor]['myosin'] = 0
                            G[node+basal_offset][neighbor+basal_offset]['myosin'] = 0
                            
                            # blacklist.append([min(node, neighbor), max(node, neighbor)])
                            
                            circum_sorted, triangles = topological_mesh(G, belt, centers)
                            circum_sorted = List(circum_sorted)

                            intercalation_callback(node,neighbor)
                            node-=1
                            break
                    j += 1
            node += 1

    pos = np.array([*nx.get_node_attributes(G,'pos').values()])
    vols = compute_cell_volumes(pos)
    PI=compute_pressure(vols)
    dists, drx = compute_edge_distance_and_direction(pos, edges, dists, drx)
    # compute_forces_vectorized(pos, PI, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)
    forces=compute_forces2(pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)
    h=np.min((timestep_bound(forces,drx,edges),1.0))
    update_pos(h, 2*h, pos, vols, edges, dists, drx, myosin, l_rest, circum_sorted, triangles, centers)
    return integrate

