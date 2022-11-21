import collections
import signal
import atexit
import os
import pickle, copy
import sys
import time
from typing import Iterable

import networkx as nx
import numpy as np

from VertexTissue.forces_orig import compute_forces_orig


from .Tissue import get_triangles, get_outer_belt, new_topology
from . TissueForces import TissueForces
from . import globals as const
from .Geometry import *
from .util import get_edges_array, set_edge_attributes, get_edge_attribute_array, set_node_attributes, get_node_attribute_array, get_node_attribute_dict
try:
    from .Player import pickle_player
    from . PyQtViz import edge_viewer
except:
    pass



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

# basal_offset=const.basal_offset
kc=const.kc



def monolayer_integrator(G, G_apical=None,
                     pre_callback=None, post_callback=None, intercalation_callback=None, termination_callback=None,
                     ndim=3, player=False, viewer=False,  save_pattern = const.save_pattern, save_rate=1.0, 
                     maxwell=False, adaptive=False, adaptation_rate=0.1, length_rel_tol=0.01, angle_tol=0.01, length_abs_tol=5e-2, minimal=False, blacklist=False, append_to_blacklist=True,
                     maxwell_nonlin=None, rest_length_func=None, RK=1, AB=1):

    if G_apical==None:
        G_apical=G

    if pre_callback is None or not callable(pre_callback):
        pre_callback = lambda t, f : None

    if post_callback is None or not callable(post_callback):
        post_callback = lambda t, f : None

    if intercalation_callback is None or not callable(intercalation_callback):
        intercalation_callback = lambda t, f : None

    force_dict = {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 
    dists = {node: 0 for node in G.edges()} 
    drx ={node: np.zeros(ndim ,dtype=float) for node in G.edges()}

    if blacklist and not isinstance(blacklist, Iterable):
        blacklist = [] 

    view = None

    pos = None
    pos_apical = nx.get_node_attributes(G_apical ,'pos')

    has_basal = 'basal_offset' in G.graph.keys()
    if has_basal:
       basal_offset=G.graph['basal_offset']

    if 'circum_sorted' in G.graph.keys():
        circum_sorted = G.graph['circum_sorted']
    else:
        circum_sorted = None

    
    if 'centers' in G.graph.keys():
        centers = G.graph['centers']
        G_centerless = G_apical.copy()
        G_centerless.remove_nodes_from(centers)
    else:
        centers = None

    belt = get_outer_belt(G_apical)
    triangles = get_triangles(G_apical, pos_apical, centers, belt)

    if 'num_apical_nodes' in G.graph.keys():
        num_api_nodes=G.graph['num_apical_nodes']
    else:
        num_api_nodes=0
    
    # @jit(nopython=True, cache=True)

    compute_forces, compute_distances_and_directions = TissueForces(G, ndim=ndim, minimal=minimal)
        

    if rest_length_func is not None:
        def get_rest_lengths(ell,L0):
                return rest_length_func(ell,L0)
    else:
        def get_rest_lengths(ell,L0):
                return L0

    ############# CONSTANTS #############
    pi_by_4=np.pi/4
    two_thirds=2/3
    one_quarter=1/4
    three_quarters=3/4
    one_sixth=1/6
    ############# END CONSTANTS #############

    def RK_effective_force(h, pos, edges, forces, l_rest, myosin):
        """
        Implements 2,3,4-order Ralston RK methods
        """
        k1 = (h/const.eta)*forces

        if RK==2:
            pos_2 = pos + two_thirds*k1
            dists_2, drx_2  = compute_distances_and_directions(pos_2, edges)
            l_rest_2 = get_rest_lengths(dists_2, l_rest)
            forces_2 = compute_forces(l_rest, dists_2, drx_2, myosin, edges, pos_2)
            return one_quarter*forces + three_quarters*forces_2

        if RK==3:
            pos_2 = pos + 0.5*k1
            dists_2, drx_2  = compute_distances_and_directions(pos_2, edges)
            l_rest_2 = get_rest_lengths(dists_2, l_rest)
            forces_12 = compute_forces(l_rest_2, dists_2, drx_2, myosin, edges, pos_2)
            k2 = (h/const.eta)*forces_12

            pos_3 = pos - k1 + 2*k2
            dists_3, drx_3  = compute_distances_and_directions(pos_3, edges)
            l_rest_3 = get_rest_lengths(dists_3, l_rest)
            forces_3 = compute_forces(l_rest_3, dists_3, drx_3, myosin, edges, pos_3)

            return one_sixth*(forces+4*forces_12+forces_3)

        if RK==4:
            pos_2 = pos + .4*k1
            dists_2, drx_2  = compute_distances_and_directions(pos_2, edges)
            l_rest_2 = get_rest_lengths(dists_2, l_rest)
            forces_2 = compute_forces(l_rest_2, dists_2, drx_2, myosin, edges, pos_2)
            k2 = (h/const.eta)*forces_2

            pos_3 = pos + .29697761*k1 + .15875964*k2
            dists_3, drx_3  = compute_distances_and_directions(pos_3, edges)
            l_rest_3 = get_rest_lengths(dists_3, l_rest)
            forces_3 = compute_forces(l_rest_3, dists_3, drx_3, myosin, edges, pos_3)
            k3 = (h/const.eta)*forces_3

            pos_4 = pos + .21810040*k1  - 3.05096516*k2  + 3.83286476*k3
            dists_4, drx_4  = compute_distances_and_directions(pos_4, edges)
            l_rest_4 = get_rest_lengths(dists_4, l_rest)
            forces_4 = compute_forces(l_rest_4, dists_4, drx_4, myosin, edges, pos_4)
  

            return .17476028*forces  - .55148066*forces_2 + 1.20553560*forces_3 + .17118478*forces_4


    def init_force_dict():
        return {node: np.zeros(ndim ,dtype=float) for node in G.nodes()} 


    edges=None

    # @profile
    def integrate(dt, t_final, 
                dt_init=None, dt_min = None, t=0, adaptive=adaptive, adaptation_rate=adaptation_rate,  verbose=False,
                angle_tol=angle_tol, length_abs_tol=length_abs_tol, length_rel_tol=length_rel_tol, append_to_blacklist=append_to_blacklist, timestep_func=None,
                pre_callback=pre_callback, post_callback=post_callback, termination_callback=termination_callback,
                save_pattern = save_pattern, save_rate=save_rate, resume=False, save_on_interrupt=False,
                orig_forces=False, check_forces=False,
                **kw):
                
        nonlocal G, G_apical, centers,  force_dict, drx, dists, view, pos, edges

        #######################################
        #           INITIALIZATION
        #######################################

        if save_pattern is not None:
            save_dict=save_pattern.find('*')==-1
        else:
            save_dict = False

        if save_dict:
            save_dict={}

        def handle_exit(*args):
            nonlocal save_dict

            # print(f'handling exit {save_dict} {args}')

            if viewer and view:
                view.kill()

            if save_dict is not False:
                print(f'saving: {save_pattern}')
                with open(save_pattern, 'wb') as file:
                    pickle.dump(save_dict, file)
                    file.close()
                print(f'done writing: {save_pattern}')
                save_dict=False

            return args

        def handle_interrupt(*args):
            handle_exit()   
            # import multiprocessing
            # multiprocessing.current_process().close()    
            sys.exit()



        if save_on_interrupt:
            # atexit.register(handle_exit)
            # signal.signal(signal.SIGKILL, silly)
            signal.signal(signal.SIGTERM, handle_interrupt)
            signal.signal(signal.SIGINT,handle_interrupt)

        def save():
            if not viewer:
                set_edge_attributes(G,'l_rest', L0)
            if save_dict is False:
                file_name = save_pattern.replace('*',str(t))
                with open(file_name, 'wb') as file:
                    pickle.dump(G, file)
            else:
                save_dict[t]=copy.deepcopy(G)

        if dt_init is None:
            h=dt
        else:
            h=dt_init

        if adaptive and dt_min is None:
            dt_min = 0

        #######################################
        #           END INITIALIZATION
        #######################################


        @jit(nopython=True, cache=True)
        def timestep_bound(forces, drxs, dists, edges, t):
            """
            Compute an upper bound on the timestep so that our tolerances are met
            """
            rad_tol = pi_by_4*angle_tol
            dt_next=dt
            for i in range(edges.shape[0]):
                df=forces[edges[i,0]]-forces[edges[i,1]]
                # df_tot = np.linalg.norm(df)
                df_radial_signed = np.sum(df*drxs[i])
                df_radial = abs(df_radial_signed)
                df_angular = np.linalg.norm(df-df*drxs[i])

                dt_radial_abs = const.eta*length_abs_tol/(df_radial+1e-12)
                dt_radial_rel = const.eta*length_rel_tol*dists[i]/(df_radial+1e-12)
                # dt_radial = 2*const.eta*length_prec*dists[i]/((df_radial+1e-12)*(2-length_prec))
                # dt_angular =  2*const.eta*rad_tol*dists[i]/(abs(2*df_angular-df_radial_signed*angle_prec+1e-12))
                dt_angular =  const.eta*rad_tol*dists[i]/(abs(df_angular+1e-12))
                dt_geom = min(dt_radial_rel, dt_radial_abs, dt_angular)

                # dt_geom = dt_radial
                dt_next = min(dt_geom, dt_next)

            return max(dt_next, dt_min)

        if timestep_func:
            timestep_bound = timestep_func




        def handle_AB():
            """Implements 2,3-Step Adams-Bashforth Multistep Integrators"""
            if AB==2:
                return (f_eff*(h+2*h_prev)-h*f_prev)/(2*h_prev)
            elif AB==3:
                return (f_eff*(6+(h/h_prev)*(2*h+6*h_prev+3*h_prev_prev)/(h_prev+h_prev_prev))
                        -f_prev*((h/h_prev)*(2*h+3*h_prev+3*h_prev_prev)/(h_prev_prev))
                        + f_prev_prev*((h/h_prev_prev)*(2*h+3*h_prev)/(h_prev+h_prev_prev)))/6

        
        # num_inter = 0 
        if (player or save_pattern is not None) and len(os.path.split(save_pattern)[0])>1:
            save_path, pattern = os.path.split(save_pattern)
            if len(save_path)>1 and not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path='.'
   
 
        pre_callback(t, force_dict)


        if player:
            pickle_player(path=save_path, pattern=pattern, start_time=t, attr='myosin', cell_edges_only=True, apical_only=True, parallel=False, save_dict=save_dict)

        if viewer:
            kwv = {'attr':'myosin', 'cell_edges_only':True, 'apical_only':True}
            if isinstance(viewer, dict):
                kwv={**kwv, **viewer}
            view = edge_viewer(G,**kwv)

        t0 = time.time()

        force_dict = init_force_dict() 

        # forces = init_force_vec()
        


        pre_callback(t, force_dict)
        

        ############# PRECOMPUTE FORCES FOR FIRST UPDATE ############

        pos=get_node_attribute_array(G,'pos')

        edges = get_edges_array(G)

        dists, drx  = compute_distances_and_directions(pos, edges)
        L0 = get_edge_attribute_array(G, 'l_rest')
        
        l_rest = get_rest_lengths(dists, L0)
        myosin = get_edge_attribute_array(G, 'myosin')

        if maxwell:
            tau = get_edge_attribute_array(G, 'tau')
            dynamic_L0 = np.isfinite(tau)

        forces = compute_forces(l_rest, dists, drx, myosin, edges, pos)


        if RK>1:
            f_eff = RK_effective_force(h, pos, edges, forces, l_rest, myosin)
        else:
            f_eff = forces

        if AB>1:
            h_prev_prev=h
            h_prev=h
            f_prev = f_eff
            f_ab = f_eff


        post_event = post_callback(t, force_dict)
        pre_event=False

        terminate = False
        if termination_callback is not None:
            terminate=termination_callback(t)
        # try:

        if save_pattern and save_rate>=0:
            print(f'saving at t={t}, in file {save_pattern}')
            t_last_save = t
            save()
        #######################################
        #              MAIN LOOP
        #######################################
        while t <= t_final and not terminate:

            ######################### UPDATE POSITIONS ######################
            ### Looks like Euler, but may not be depending on RK keyword ####

            if AB==1:
                pos[:,:ndim] += (h/const.eta)*f_eff
            else:
                pos[:,:ndim] += (h/const.eta)*f_ab

                
            ############## UPDATE NETWORK OBJECT ###########################
            set_node_attributes(G,'pos',pos)

            ############ PRECOMPUTE DISTANCES SO WE CAN USE Crank-Nicolson FOR REST LENGTH UPDATES #########
            dists_prev=dists
            dists, drx = compute_distances_and_directions(pos, edges)

            ################### UPDATE L0 ##############################
            if maxwell or maxwell_nonlin:
                
                
                
                
                if not maxwell_nonlin:
                    r = h/(tau*2)
                    L0[dynamic_L0] = ((L0*(1-r)+r*(dists+dists_prev))/(1+r))[dynamic_L0]
                else:
                    L0[dynamic_L0] += (h/2)*((maxwell_nonlin(dists, L0)+maxwell_nonlin(dists_prev,L0))/tau)[dynamic_L0]

                if viewer:
                    set_edge_attributes(G,'l_rest', L0)


            intercalation=check_for_intercalations(t, append_to_blacklist=append_to_blacklist)


            ############ INCREMENT TIME ############
            t = t + h

            if verbose:
                t1=time.time()
                print(f'{t1-t0:.12f} seconds elapsed, h={h:.12f}, t={t:.12f}') 
                t0 = t1

            force_dict = init_force_dict()
            pre_event = pre_callback(t, force_dict)
            




            if AB==2:
                f_prev = f_eff
            elif AB==3:
                f_prev_prev = f_prev
                f_prev = f_eff

            if pre_event or post_event or intercalation:
                L0 = get_edge_attribute_array(G, 'l_rest')
                myosin = get_edge_attribute_array(G, 'myosin')

            l_rest = get_rest_lengths(dists, L0)
            forces = compute_forces(l_rest, dists, drx, myosin, edges, pos, recompute_indices=intercalation)





            #     f_ab=f_eff

            if RK>1:
                f_eff = RK_effective_force(h, pos, edges, forces, l_rest, myosin)
            else:
                f_eff = forces


            if AB>1:
                f_ab=handle_AB()
                
            if orig_forces:
                force_orig=compute_forces_orig(G)
                if check_forces:
                    print(f'max error: {np.abs((f_eff-force_orig)).max()}')
                f_eff = force_orig

            if adaptive:
                if AB>1:
                    h_prev_prev=h_prev
                    h_prev=h
                    f_pred=f_ab
                else:
                    f_pred=f_eff

                hnew = timestep_bound(f_pred, drx, dists , edges, t)


                if hnew>h:
                    h=(1-adaptation_rate)*h+adaptation_rate*hnew
                else:
                    h=hnew

                if RK>1:
                    f_eff = RK_effective_force(h,pos, edges, forces, l_rest, myosin)
                    hbound = timestep_bound(f_eff, drx, dists , edges, t)
                    while hbound<h:

                        h=max(h/2, dt_min)
                        if h==dt_min:
                            break
                        f_eff = RK_effective_force(h , pos, edges, forces, l_rest, myosin)
                        hbound = timestep_bound(f_eff, drx, dists , edges,t)
                        

                if AB>1:
                    f_ab=handle_AB()
                    hbound = timestep_bound(f_ab, drx, dists , edges, t)
                    while hbound<h:

                        h=max(h/2, dt_min)
                        if h==dt_min:
                            break
                        f_ab =handle_AB()
                        hbound = timestep_bound(f_ab, drx, dists , edges, t)

            else:
                h=dt
                
            post_event = post_callback(t, force_dict)

            if save_pattern and save_rate>=0:
                delta_save = (t - t_last_save) - save_rate
                if intercalation or (delta_save >= 0 or abs(delta_save)  <= h/2): 
                    t_last_save = t
                    save()

            if viewer:
                view(G, title = f't={t}')

            if termination_callback is not None:
                terminate=termination_callback(t)

        

            
        # set_edge_attributes(G,'l_rest', L0)
        if save_pattern and (save_rate<0 or terminate) : 
            save()
            
        handle_exit()
        
    
    # @profile
    l_mvmt = const.l_mvmt
    blacklisting = isinstance(blacklist, collections.abc.Iterable)
    def check_for_intercalations(t, append_to_blacklist=True):
        nonlocal circum_sorted, G_apical, G_centerless, blacklist, view, dists, drx, edges, pos


        if dists.min() > const.l_intercalation:
            return False

        pos_dict = get_node_attribute_dict(G,'pos')
        # node=0
        intercal = False
        nodes = list(G_centerless.nodes)
        num_api_nodes = len(nodes)
        # print('checking')
        i=0
        
        while i<num_api_nodes:
            node=nodes[i]
        # for node in range(0,num_api_nodes):
            if node not in belt: 
                nhbrs=list(G_centerless.neighbors(node))
                j=0
                while j<len(nhbrs):
                    neighbor=nhbrs[j] #and (neighbor not in belt) 
                    if (not blacklisting or (min(neighbor,node), max(neighbor,node)) not in blacklist): 
                    
                        a = pos_dict[node]
                        b = pos_dict[neighbor]
                        if has_basal:
                            c = pos_dict[node+basal_offset]
                            d = pos_dict[neighbor+basal_offset]
                        
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
                                if has_basal:
                                    avg_loc = (np.array(c) + np.array(d)) / 2.0 
                                    c = avg_loc 
                                    d = avg_loc 
                                # move nodes toward new center 
                                # apical 
                                cents = list(set(G.neighbors(node)) & set(G.neighbors(neighbor)))
                                mvmt = unit_vector(a,pos_dict[cents[1]])
                                a = np.array([a[0]+l_mvmt*mvmt[0], a[1]+l_mvmt*mvmt[1], a[2]+l_mvmt*mvmt[2]])
                                G.node[node]['pos'] = a 
                                mvmt = unit_vector(b,pos_dict[cents[0]])
                                b = np.array([b[0]+l_mvmt*mvmt[0], b[1]+l_mvmt*mvmt[1], b[2]+l_mvmt*mvmt[2]])
                                G.node[neighbor]['pos'] = b 

                                pos_dict[node]=a
                                pos_dict[neighbor]=b
                                # basal 
                                #cents = list(set(G.neighbors(node+basal_offset)) & set(G.neighbors(neighbor+basal_offset)))
                                if has_basal:
                                    mvmt = unit_vector(c,pos_dict[cents[1]+basal_offset])
                                    c = np.array([c[0]+l_mvmt*mvmt[0], c[1]+l_mvmt*mvmt[1], c[2]+l_mvmt*mvmt[2]])
                                    G.node[node+basal_offset]['pos'] = c 
                                    mvmt = unit_vector(d,pos_dict[cents[0]+basal_offset])
                                    d = np.array([d[0]+l_mvmt*mvmt[0], d[1]+l_mvmt*mvmt[1], d[2]+l_mvmt*mvmt[2]])
                                    G.node[neighbor+basal_offset]['pos'] = d 
                                
                                ii = list((set(list(G.neighbors(node))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))[0]
                                jj = list((set(list(G.neighbors(neighbor))) & set(list(centers))) - (set(list(G.neighbors(node))) & set(list(G.neighbors(neighbor)))))
                                temp1 = list(set(G.neighbors(node)) & set(G.neighbors(cents[0])))
                                temp1.remove(neighbor)
                                temp2 = list(set(G.neighbors(neighbor)) & set(G.neighbors(cents[1])))
                                temp2.remove(node)

                                if has_basal:
                                    offsets=(0, basal_offset)
                                else:
                                    offsets=(0,)

                                for offset in offsets:

                                    old_rods = [(node, cents[0]), (node, temp1[0]), (neighbor, cents[1]), (neighbor, temp2[0])]
                                    new_rods=[(node, temp2[0]), (neighbor, temp1[0]), (neighbor, ii)]
                                    if len(jj):
                                        new_rods.append((node, jj[0]))
                                    old_rods = np.array(old_rods)+offset
                                    new_rods = np.array(new_rods)+offset
                                    
                                    old_attrs = [ G[s[0]][s[1]] for  s in old_rods]
                                                
                                    # # sever connections
                                    for spoke in old_rods:
                                        G.remove_edge(*spoke)
                                    
                                    # # add new connections
                                    # new edges 
                                    G.add_edge(*new_rods[0],**old_attrs[3])
                                    G.add_edge(*new_rods[1],**old_attrs[1])
                                    # new spokes
                                    G.add_edge(*new_rods[2], **G[node+offset][ii+offset])
                                    if len(jj):
                                        G.add_edge(*new_rods[3], **G[neighbor+offset][jj[0]+offset])

                                    # # reset myosin on contracted edge
                                    G[node+offset][neighbor+offset]['myosin'] = 0
                                if blacklisting and append_to_blacklist:
                                    blacklist.append((min(node, neighbor), max(node, neighbor)))
                                
                                circum_sorted, triangles, G_apical = new_topology(G_apical, 
                                                                                  [node, neighbor],
                                                                                   cents, 
                                                                                   temp1,
                                                                                   temp2, 
                                                                                   ii, 
                                                                                   jj, 
                                                                                   belt, 
                                                                                   centers, 
                                                                                   num_api_nodes,
                                                                                   adjust_network_positions=G is not G_apical)
                                G.graph['circum_sorted'] = circum_sorted
                                G.graph['triangles'] = triangles

                                G_centerless = G_apical.copy()
                                G_centerless.remove_nodes_from(centers)

                                edges = np.array([*G.edges().keys()],dtype=int)
                                pos = get_node_attribute_array(G,'pos')

                                dists, drx = compute_distances_and_directions(pos, edges)
                                intercalation_callback(node, neighbor)
                                i-=1
                                if viewer:
                                    view(G, title = f'freshly resolved intercalation t={t}')
                                    print('resolved')
                                break
                    j += 1

            i +=1
        return intercal
    return integrate

