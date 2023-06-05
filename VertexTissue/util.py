
from audioop import cross
from cmath import pi
from functools import partial
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from collections.abc import Iterable

import __main__
from VertexTissue.Filesystem import get_filenames


from VertexTissue.Geometry import euclidean_distance, unit_vector
from VertexTissue.Iterable import imin



def new_graph(G=None):
    if G is None:
        G = nx.Graph()
    else:
        G = G.copy()
    if nx.__version__ > "2.3":
        G.node = G._node

    return G


def get_points(G, q, pos):
    # get node numbers associated with a given center
    # inputs:   G: networkx graph
    #           q: number of center node (apical only)
    #           pos: position of nodes 
    # returns:  pts: list of positions that are associated with that center 
    basal_offset=G.graph['basal_offset']
    api_nodes = [q] + list(G.neighbors(q))
    basal_nodes = [q+basal_offset] + list(G.neighbors(q+basal_offset)) 
#    basal_nodes = [api_nodes[n] + 1000 for n in range(1,7)]
    pts = api_nodes + basal_nodes
    pts = [pos[n] for n in pts]

    return pts 

def find_first(x):
    return np.argwhere(x)[0,0]


def first(bools):
    return np.argwhere(bools)[0]


def polygon_area(x, y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def rectify_network_positions(G, phi=0):
    pos_dict = nx.get_node_attributes(G, 'pos')
    pos_dict = {k: v[0:2] for k, v in pos_dict.items()}

    arr = np.array([*pos_dict.values()])
    arr -= arr[0]
    theta = np.arccos(np.dot(unit_vector(arr[0], arr[3]), (1, 0)))

    R = rotation_matrix(np.pi/2-theta+phi)

    return {k: np.matmul(R, v) for k, v in pos_dict.items()}

def has_basal(G):
    return 'basal_offset' in G.graph.keys()



def get_cell_edges(G, basal=False, excluded_nodes=[]):

    centers = G.graph['centers']
    exclude=[*excluded_nodes, *centers]
    if not basal and has_basal(G):
        basal_offset=G.graph['basal_offset']
        return [(a, b) for a, b in G.edges if a<=basal_offset and b<=basal_offset and (a not in exclude) and (b not in exclude)]
    else:
        return  [(a, b) for a, b in G.edges if (a not in exclude) and (b not in exclude)]
    
def get_spokes(G, basal=True):

    centers = [*G.graph['centers']]

    if not basal and has_basal(G):
        basal_offset=G.graph['basal_offset']
        return [(a, b) for a, b in G.edges if a<=basal_offset and b<=basal_offset and (a in centers) or (b in centers)]
    else:
        return  [(a, b) for a, b in G.edges if (a in centers) or (b  in centers)]

def get_myosin_free_cell_edges(G, basal=False, excluded_nodes=[]):
    edges = get_cell_edges(G, basal=basal, excluded_nodes=excluded_nodes)
    return [(a, b) for a, b in edges if G[a][b]['myosin'] == 0]

def shortest_edge_network_and_time(d, excluded_nodes=[], return_length=False):
    min_lens = []
    
    times = list(d.keys())
    G = d[times[-2]]

    edges = get_myosin_free_cell_edges(G, excluded_nodes=excluded_nodes)

    for t in times:
        G = d[t]
        lens = [euclidean_distance(G.nodes[a]['pos'], G.nodes[b]['pos']) for a, b in edges]
        min_lens.append(min(lens))

    # edge_view(G)
    ind_min = imin(min_lens)
    t_min = times[ind_min]
    G = d[t_min]
    if not return_length:
        return G, t_min
    else:
        return G, t_min, min_lens[ind_min]

def shortest_edge_length_and_time(d, excluded_nodes=[]):
    _, t_min, l_min = shortest_edge_network_and_time(d, excluded_nodes=excluded_nodes, return_length=True)
    
    
    # edges = [(a, b) for a, b in G.edges if (a not in exclude)  and (b not in exclude) and G[a][b]['myosin'] == 0]
    # nodes = G.nodes
    # lens = [euclidean_distance(nodes[a]['pos'], nodes[b]['pos']) for a, b in edges]

    return l_min, t_min

def arc_to_edges(*args, sort=True):
    edges=[[tuple((x[i-1], x[i])) for i in range(len(x))] for x in args]
    edges = [y for x in edges for y in x]

    if sort:
        edges=[tuple(sorted(e)) for e in edges]

    return edges
        
    
def edge_index(G, e):
    if len(e):
        edges = get_edges_array(G)

        def find_edge(e):
            return np.argwhere(np.logical_or(np.all(e == edges, axis=1), np.all(e == np.fliplr(edges), axis=1)))[0,0]
        
        if isinstance(e[0], Iterable) and len(e[0])==2:
            return np.array([ find_edge(_) for _ in e])
        else:
            return find_edge(e)
    else:
        return []

def get_node_attribute_dict(G, attr):
    return nx.get_node_attributes(G, attr)


def get_node_attribute_array(G, attr):
    return np.array((*nx.get_node_attributes(G, attr).values(),))


def get_edge_attribute_array(G, attr, dtype=float):
    return np.fromiter(nx.get_edge_attributes(G, attr).values(), dtype=dtype)

def set_dict_values_from_edge_attrs(G, attr, dtype=float, d=None):
    if d is None:
        return  set_dict_values_from_edge_attrs(G, attr, dtype=dtype, d={})
    else:
        arr = get_edge_attribute_array(G, attr, dtype=dtype)
        if len(arr):
            d[attr] = arr

def set_node_attributes(G, attr, vals):
    for k, v in enumerate(vals):
        G.node[k][attr] = v

def get_edges_array(G):
    return np.array([*G.edges().keys()], dtype=int)




def set_edge_attributes(G, attr, vals):
    attr_vals = {e: v for e, v in zip(G.edges(), vals)}
    nx.set_edge_attributes(G, attr_vals, name=attr)

def inside_arc(node, arc, G):
    if node in arc:
        return False
        
    closest = np.argmin([nx.shortest_path_length(G, source=node, target=a) for a in arc])
    return nx.shortest_path_length(G, source=0, target=node) < nx.shortest_path_length(G, source=0, target=arc[closest])


def inside_radius(node, radius, G):
    center_xy = G.node[0]['pos'][:2]
    node_xy = G.node[node]['pos'][:2]
    return euclidean_distance(center_xy, node_xy) < radius
  
def crossing_edges(radius, G, basal=False):
    edges = get_cell_edges(G, basal=basal)
    return [e for e in edges if inside_radius(e[0],radius,G) is not inside_radius(e[1],radius,G) ]

def circumferential_arc(radius, G, continuous=True):

    arc=[]

    def link_to_next(nodes):
        nonlocal arc

        path_lengths = [nx.shortest_path_length(G_edge, source=arc[-1], target=n) for n in nodes]
        path_lengths_reversed = [nx.shortest_path_length(G_edge, source=arc[0], target=n) for n in nodes]

        if min(path_lengths)>min(path_lengths_reversed):
            arc = list(reversed(arc))
            next_node = nodes[imin(path_lengths_reversed)]
        else:
            next_node = nodes[imin(path_lengths)]
            
        link = nx.shortest_path(G_edge, source=arc[-1], target=next_node)

        arc.extend(link[1:-1]) #add in any intermediate node
        return next_node

    edges = get_cell_edges(G)
    G_edge = new_graph()
    G_edge.add_edges_from(edges)
    crossers=[e for e in edges if inside_radius(e[0],radius,G) is not inside_radius(e[1],radius,G) ]
    
    G_edge.remove_edge(*crossers[0])
    arc.extend([*crossers.pop(0)])

    if not continuous:
        for e in crossers:
            arc.extend(e)
        return arc

    while len(crossers)>0:
        connected=False
        for i in range(len(crossers)):
            if edge_contains_node(crossers[i], arc[-1]):
                connected=True
                arc.append(sorted(crossers[i], key=lambda x: abs(x-arc[-1]))[-1])
                G_edge.remove_edge(*crossers[i])
                crossers.pop(i)
                break

        if not connected and continuous:

            targets =  [n for n in np.unique(np.array(crossers).ravel()) if n not in arc]

            next_node = link_to_next(targets)

            for i in range(len(crossers)):
                if edge_contains_node(crossers[i], next_node):
                    connected=True
                    new_nodes=sorted(crossers[i], key=lambda x: abs(x-next_node))
                    arc.extend(new_nodes)
                    G_edge.remove_edge(*crossers[i])
                    crossers.pop(i)
                    break
                
        if not connected and continuous:
            print('uh oh dawg')

    if arc[-1] not in G_edge.neighbors(arc[0]) and continuous:
        arc.extend(nx.shortest_path(G_edge, source=arc[-1], target=arc[0])[1:-1])
    return arc

def adjacent_edges(a,b):
    return (a[0]==b[0] or a[0]==b[1] or a[1]==b[0] or a[1]==b[1]) and not same_edge(a,b)

def same_edge(a,b):
   return ((a[0]==b[0] and a[1]==b[1]) or (a[1]==b[0] and a[0]==b[1]))

def edge_contains_node(e,n):
    return e[0]==n or e[1]==n 

def pcolor(X, Y, C, shading='nearest', cmap=None, tight=True, **kw):
    plt.pcolormesh(X, Y, C, shading=shading, cmap=cmap, **kw)
    if tight:
        plt.xlim(X[0], X[-1])
        plt.ylim(Y[0], Y[-1])

def hatched_contour(X, Y, Z, hatches = ['\\\\\\'], color='#ff000080', upscale=1, levels = None, linewidth=0.5, alpha=0.0, hatch_alpha=0.05):
    if upscale>1:
        XY, Z = upsample(X, Y, Z, fold=upscale, midpoint=True)

    color = matplotlib.colors.to_rgba(color)
    if  hatch_alpha is not None:
        color = (*color[:3], hatch_alpha)
    color = matplotlib.colors.to_hex(color, keep_alpha=True)
    plt.rcParams['hatch.color']=color
    plt.rcParams['hatch.linewidth']=linewidth


    plt.contourf(*XY, Z, levels=levels, hatches=hatches, alpha=alpha)
    plt.contour(*XY, Z, levels =levels, colors=matplotlib.colors.to_hex(color, keep_alpha=False), linewidths=linewidth)

def contour(X, Y, Z, color='r', upscale=1, levels = None, linewidth=1):
    if upscale>1:
        XY, Z = upsample(X, Y, Z, fold=upscale, midpoint=True)

    plt.contour(*XY, Z, levels =levels, colors=color, linewidths=linewidth)

def upsample(*args, fold=10, midpoint=True):
    *coords, vals = args 
    ndim = len(vals.shape)

    for i, x in enumerate(coords):
        Nx=len(x)
        coords
        x0=np.array(x)
        if midpoint:
            x = np.array([x0[0], *((x0[1:]+x0[:-1])/2), x0[-1]])
            X = np.array([*np.linspace(0, 1, fold), *np.linspace(1,Nx-1,fold*(Nx-2)+2)[1:-1], *np.linspace(Nx-1,Nx,fold) ])
        else:
            X=np.linspace(0, Nx-1, fold*Nx)
        coords[i] = np.interp(X, range(len(x)), x)


    Z = vals
    
    for d in range(ndim):
        Z = np.repeat(Z, fold, axis=d)


    return coords, Z


def collate_pickles(path='.',pattern='*', step=1):
    file_and_time_list = get_filenames(path=path, pattern=pattern, include_path=True)
    d={}
    for i in range(0,len(file_and_time_list), step):
        f,t = file_and_time_list[i]
        with open(f,'rb') as file:
            obj = pickle.load(file)
            d[t]=obj
    return d

def finder(a, b):
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0])
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val]-b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt == True)[0][bw]
    return np.column_stack((uni, ret_b))

class conditional_decorator(object):
    def __init__(self, dec, condition, **kw):
        self.decorator = dec
        self.condition = condition
        self.kw = kw

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func, **self.kw)
# def conditional_decorator(dec, condition, **kw):
#     def decorator(func):
#         if not condition:
#             # Return the function unchanged, not decorated.
#             return func
#         return partial(dec, **kw)(func)
#     return decorator

