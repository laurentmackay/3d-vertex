import sys
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
    with os.scandir(path) as it:
        for entry in it:
            name = entry.name
            match = regex.match(name)

            if  match and entry.stat().st_ctime > min_timestamp:
                start, end = match.regs[1]
                try:
                    time = float(name[start:end])
                    if include_path:
                        name =os.path.join(path, name)
                    out.append((name, time))
                except:
                    pass
                # print(entry.name)

    out.sort(key=lambda x: x[1])
    if extend:
        extend.extend(out)
    return out

def get_creationtime(filename, path=os.getcwd()):
    return os.stat(os.path.join(path, filename)).st_ctime

def np_find(arr, x):
    np.argwhere([ np.all(e == x)for e in arr ])