import pickle, os
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np

from .util import get_creationtime, get_filenames


def analyze_network_evolution(path='.', start_time=0, pattern=None, func=lambda G, t: None, processes=None):
    try:
        def load_and_run(a):
            with open(a[0],'rb') as file:
                G = pickle.load(file)
                return (a[1], func(G,a[1]))

        start_time=0
        start_file = pattern.replace('*',str(start_time))
        start_timestamp = get_creationtime( start_file, path=path)
        start_file = os.path.join(path, start_file)
        
        file_list = [(start_file, start_time)]
        get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp, extend=file_list, include_path=True)

        pool = Pool(processes)
        return pool.map(load_and_run, file_list)
    except:
        return None

def analyze_networks(path='.', patterns=None, func=lambda G, t: None, processes=None):
    return [ analyze_network_evolution(path=path, pattern=p, func=func, processes=processes ) for p in patterns ]


