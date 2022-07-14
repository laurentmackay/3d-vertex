import pickle
import sys, os, inspect, __main__

from VertexTissue.funcs import euclidean_distance, unit_vector
IS_WINDOWS = sys.platform.startswith('win')

import multiprocessing as mp
import os

import fnmatch, re

import networkx as nx
import numpy as np

from . import globals as const


def new_graph(G=None):
    if G is None:
        G = nx.Graph()
    else:
        G=G.copy()
    if nx.__version__>"2.3":
        G.node=G._node
    
    return G



if IS_WINDOWS:
    import dill
    def run_dill(payload, args):
        fun = dill.loads(payload)
        return fun(*args)

    def make_dill(f,a):
        return (dill.dumps(f),(a,))

def mkprocess(target, args=tuple() ,daemon=True):
    if IS_WINDOWS:
        proc = mp.Process(target=run_dill, args=make_dill(target, *args), daemon=daemon)
    else:
        proc = mp.Process(target=target, args=args, daemon=daemon)
    return proc

def get_filenames(path='.', pattern=const.save_pattern, min_timestamp=0, extend=None, include_path=False):
    out=[]
    pattern = fnmatch.translate(pattern)
    pattern = pattern.replace('.*','(.*)')
    regex = re.compile(pattern)
    if extend is not None:
        filenames = [ e[0] for e in extend]
    with os.scandir(path) as iter:
        for entry in iter:
            name = entry.name
            match = regex.match(name)

            if  match and entry.stat().st_ctime > min_timestamp:
                start, end = match.regs[1]
                try:
                    time = float(name[start:end])
                    if include_path:
                        name =os.path.join(path, name)
                    if extend is None or name not in filenames:
                        out.append((name, time))
                except:
                    pass


    out.sort(key=lambda x: x[1])
    if extend:
        extend.extend(out)
    return out

def get_creationtime(filename, path=os.getcwd()):
    return os.stat(os.path.join(path, filename)).st_ctime

def get_oldestfile(pattern, path=os.getcwd()):
    out=[]
    pattern = fnmatch.translate(pattern)
    pattern = pattern.replace('.*','(.*)')
    regex = re.compile(pattern)
    min_timestamp = np.inf
    with os.scandir(path) as iter:
        for entry in iter:
            name = entry.name
            # print(name)
            match = regex.match(name)

            if  match:
                creation_time = entry.stat().st_ctime 
                if creation_time < min_timestamp:
                    start, end = match.regs[1]
                    try:
                        time = float(name[start:end])

                        out=(name, time)
                    except:
                        pass



    return out

def np_find(arr, x):
    np.argwhere([ np.all(e == x)for e in arr ])

def first(bools):
    return np.argwhere(bools)[0]


def polygon_area(x,y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1]* x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5*np.abs(main_area + correction)
    
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def rectify_network_positions(G, phi=0):
    pos_dict=nx.get_node_attributes(G,'pos')
    pos_dict = {k:v[0:2] for k,v in pos_dict.items()}

    arr=np.array([*pos_dict.values()])
    arr-=arr[0]
    theta = np.arccos(np.dot(unit_vector(arr[0],arr[3]),(1,0) ))

    R=rotation_matrix(np.pi/2-theta+phi)

    return {k:np.matmul(R, v) for k,v in pos_dict.items()}

def last_item(d):
    return d[next(reversed(d.keys()))]


def imin(iter):
    return min(range(len(iter)), key=iter.__getitem__)

def imax(iter):
    return max(range(len(iter)), key=iter.__getitem__)


def shortest_edge_network_and_time(d):
    min_lens = []

    times=list(d.keys())
    G=d[times[1]]
    centers = G.graph['centers']
    edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]
    for t in times:
        G=d[t]
        lens = [ euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos']) for a,b in edges]
        min_lens.append(min(lens))

    
    # edge_view(G)
    t_min=times[imin(min_lens)]
    G=d[t_min]

    return G, t_min

def shortest_edge_length_and_time(d):
    G, t_min = shortest_edge_network_and_time(d)
    centers = G.graph['centers']
    edges = [(a,b) for a,b in G.edges if (a not in centers ) and (b not in centers) and G[a][b]['myosin']==0 ]

    lens = [ euclidean_distance(G.nodes[a]['pos'],G.nodes[b]['pos']) for a,b in edges]

    return min(lens), t_min

def signature_string(f=None, locals=None):

    stack=inspect.stack()
    if f is None or locals is None:
        caller=stack[1]
        if f is None:
            f=caller.frame.f_globals[caller.function]
        if locals is None:
            locals=caller.frame.f_locals
            
    sig=inspect.signature(f)
    
    args, kws = signature_lists(f)

    arg_strs = [str(locals[p]) for p in args]

    kw_strs=[]
    
    for p in kws:
        if p in locals.keys():
                val=locals[p]
                default = sig.parameters[p].default ;
                if default != val:
                    if not isinstance(default, bool):
                        kw_strs.append(p+'='+str(locals[p]))
                    elif default==True:
                        kw_strs.append('no_'+p)
                    else:
                        kw_strs.append(p)

            

    out=''

    if len(kw_strs):
        out += '_'.join(kw_strs)
    
    if len(arg_strs):
        new = ','.join(arg_strs)
        if len(out):
            out+='_'+new
        else:
            out=new
    elif len(out)==0:
        out='_'

    return  out


def signature_lists(f):

    sig=inspect.signature(f)

    args=[]
    kws=[]
    
    for p in sig.parameters:
        if sig.parameters[p].default is not inspect.Parameter.empty:
            kws.append(p)          
        else:
            args.append(p)

    

    return  args, kws

def check_funcion_cache(func):
    cache_folder=os.path.join('.cache',script_name(), filename_without_extension(func.__code__.co_filename))
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)



    cache=os.path.join(cache_folder, func.__name__)

    if os.path.exists(cache):
        with open(cache, 'rb') as file:
            results = pickle.load(file)

    else:
        results=None

    return results, cache


def script_name():
    return  filename_without_extension(sys.argv[0])

def filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]