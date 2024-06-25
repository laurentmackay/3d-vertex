
import pickle, os, collections, psutil, multiprocessing
# from jug import TaskGenerator
from collections.abc import Iterable
from itertools import product
from pathos.pools import ProcessPool as Pool
from pathos.pools import ThreadPool as ThreadPool
import numpy as np
import networkx as nx


from .Filesystem import   get_creationtime, get_filenames, get_oldestfile
from .Caching import  signature_string, function_savedir, cached
from .Dict import hash384

def analyze_network_evolution(path='.', start_time=0, pattern=None, func=lambda G, t: G, pool=None, processes=None, indices=None):
    try:
        def load_and_run(a):
            with open(a[0],'rb') as file:
                G = pickle.load(file)
                return (a[1], func(G,a[1]))

        start_file = pattern.replace('*',str(start_time))

        if not os.path.exists(os.path.join(path,start_file)):
            start_file, start_time = get_oldestfile(pattern, path)

        start_timestamp = get_creationtime( start_file, path=path)

        start_file = os.path.join(path, start_file)
        
        file_list = [(start_file, start_time)]
        get_filenames(path=path, pattern=pattern, min_timestamp=start_timestamp, extend=file_list, include_path=True)

        if indices is not None:
            file_list = [file_list[i] for i in indices]

        if pool is None:
            pool = Pool(processes)

        results = pool.map(load_and_run, file_list)

        lens = np.array([ len(r[1]) if isinstance(r[1], collections.Iterable) else 1.0 for r in results ])

        
        if lens.min() == lens.max() and not isinstance(results[0][1], nx.Graph):
            try:
                t = np.array([r[0] for r in results]).reshape(-1,1)
                data = np.array([np.array(r[1]) for r in results]).reshape(len(t),-1)
                results = np.hstack((t,  data))
            except:
                pass

        return results
    except:
        return None

analyze_network=analyze_network_evolution

def analyze_networks(patterns, **kw):
    pool = Pool(None)

    results=[ analyze_network_evolution(pattern=p, pool=pool, **kw) for p in patterns ]
    lens = np.array([ len(r) if isinstance(r, collections.Iterable) else 1.0 for r in results ])
    isnumpy = np.array([isinstance(r, np.ndarray) for r in results])
    if lens.min() == lens.max() and np.all(isnumpy):
        results = np.vstack(results)
    return results 


def check_function_cache(func, load=True, kw={}, pre_hash=''):
    cache = os.path.join('.cache', function_savedir(func))
    result = cached(func, load=load, filename=hash384(kw, pre_hash=pre_hash), cache_dir='.cache')
    if kw or pre_hash:
        cache = os.path.join(cache, )
    cache_dir = os.path.dirname(cache)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(cache) and load:
        with open(cache, 'rb') as file:
            results = pickle.load(file)

    else:
        results = None

    return results, cache


def parameter_sweep(params, simulations, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None, cache=None):

    results = None
    shape=(len(params), len(simulations))

    if cache is not None and pre_process is not None:
        if cache == True:
            results, cache = check_function_cache(pre_process)


        
    if results is None or not (results.shape==shape):
        results= np.array([ [None]*len(simulations) ]*len(params))

    

    if pool is None:
        core_count = psutil.cpu_count(logical=False)
        pool = Pool(nodes=core_count-1)

    
    


    def runner(k):
        i,j = np.unravel_index(k, shape)
        pars=params[i]
        simulation=simulations[j]
        path = savepaths[i][j]
        missing_file = not os.path.exists(path)

        run = (savepaths is None and inpaint is None)  or overwrite or (missing_file and inpaint is None)
        
        
        if run:
            return simulation(pars)
        else:
            return None


    inpaint_ij=multiprocessing.Manager().list()

    results_raveled = results.ravel()

    needed = np.where(results_raveled==None)[0]

    results_raveled[needed] = pool.map(runner, needed)

    new = len(needed) > 0 

    def loader(k):
        i,j = np.unravel_index(k, shape)

        path = savepaths[i][j]
        file_exists = os.path.exists(path)
        if file_exists:
            print(f'loading[{i}][{j}]: {path}  ({os.path.getsize(path)/(1024*1024)} mb)')

            with open(path, 'rb') as file:
                try:
                    result = pickle.load(file)
                except:
                    result = None

                if result is not None and pre_process is not None:
                    try:
                        result = pre_process(result)
                    except:
                        result=None
        else:
            result = None


        if result is None:
            # print(f'inpainting[{i}][{j}]: {inpaint}')
            inpaint_ij.append(k)
                

            
        return result


    
  
    
    needed = np.where(results_raveled==None)[0]
    results_raveled[needed] = pool.map(loader, needed)


    #cache if we have any new results
    if new and pre_process is not None and cache:
        with open(cache, 'wb') as file:
                pickle.dump(results, file)
    
    #inpaint after we have cached
    if inpaint is not None:
        for k in inpaint_ij:
            results_raveled[k] = inpaint





    return results



# def sweep(*args, kw={}, savepath_prefix='.', extension='.pickle', overwrite=False, 
#             pool=None, pre_process=None, pre_process_kw={}, pass_kw=False,
#             inpaint=None, cache=False, refresh=False, verbose=True):
#     """
#     Perform a sweep of a function over parameter and keyword combinations, or retrieve corresponding results from local storage.

#     Args:
#         *args (function or Iterable): The function(s) and the Iterable(s) containing parameters that the function(s) are evaluated with. 
#             If multiple parameter iterables are given, they will be expanded into a single Iterable using itertools.product. Any results
#             that are not saved to storage by the function will be automatically saved.
            
#         kw (dict or Iterable): Keyword dict(s) to be fed into the function(s). If a dict is given, all keys must be strings. 
#             If an Iterable is given, all its members must be keyword dicts.

#         inpaint (Any): If not None, do not evaluate any function(s), but replace any missing results with `inpaint`. 
#             Default is None.

#         savepath_prefix (str): The base path where results are to be saved/loaded. Default is the present working directory '.'.  

#         extension (str): The file extension for the serialized results. Default is '.pickle'.

#         overwrite (bool): Controls whether existing results are to be overwritten. Default is False.

#         pool (Mappable worker pool): The worker pool that carries out the function evaluations. If None (default), 
#             then a mulitprocessing pool with n-1 nodes is created, where n is the number of physical cores on the machine.

#         pre_process (function): A function for pre-processing results after function execution or loading from storage.
#             Default is None.

#         cache (bool): If pre_process is not None, the processed results are stored in a local cache for rapid later retrieval.
#              Default is False.

#         refresh (bool): If pre_process is not None and cache is True, ignore the current cache and overwrite it after processing. 
#             Default is False.
        
#         verbose (bool): Controls whether a message is printed when files are loaded

#     """

#     func = get_callables(*args)
#     params = tuple(product(*get_iterables(*args)))

#     if len(func) > 1:
#         for f in func:
#             sweep(f, params, savepath_prefix=savepath_prefix, extension=extension, overwrite=overwrite, pool=pool, pre_process=pre_process, inpaint=inpaint, cache=cache)
#     elif len(func) < 1:
#         raise TypeError('No function specified')
#     else:
#         func = func[0]

#     extension = extension.strip()
#     if extension[0] != '.':
#         extension = '.'+extension

#     if isinstance(kw, dict):
#         kw = [kw]

#     basepath = os.path.join(savepath_prefix, function_savepath(func))
#     results = None
#     shape = (max(len(params), 1),  len(kw))

#     if cache is not None and pre_process is not None:
#         if cache == True:
#             pre_hash = dict_hash([*params, *kw])
#             results, cache = check_function_cache(pre_process, kw=pre_process_kw, load=not refresh, pre_hash=pre_hash)

#     if results is None or not (results.shape == shape):
#         results = np.array([[None]*shape[1]]*shape[0])

#     if pool is None:
#         core_count = psutil.cpu_count(logical=False)
#         pool = Pool(nodes=core_count-1)

      
#     #setup a graceful exit
#     signal.signal(signal.SIGTERM, lambda a,b: pool.terminate())

#     results_raveled = results.ravel()

#     par_names, _ = signature_lists(func)

#     savepaths = multiprocessing.Manager().dict()

#     def check_filesystem_and_run(k):

#         i, j = np.unravel_index(k, shape)

#         pars = params[i] if params else params

#         if not isinstance(pars, Iterable):
#             pars = (pars,)
#         locals = {**{a: v for a, v in zip(par_names, pars)}, **kw[j]}

#         path = os.path.join(basepath, signature_string( f=func, locals=locals) + extension)

#         savepaths[k] = path
#         missing_file = not os.path.exists(path)
#         run = inpaint is None and (overwrite or missing_file)

#         if run:
#             result = func(*pars, **kw[j])
#             #save the result for later, if that makes sense
#             if result is not None and not os.path.exists(path):
#                 with open(path, 'wb') as file:
#                     pickle.dump(result, file)
#         else:
#             result = None

#         return result

#     to_check = np.where(results_raveled == None)[0]

#     possibly_new_results = len(to_check) > 0

#     # signal.pthread_sigmask(signal.SIG_BLOCK,[signal.SIGINT])


#     results_raveled[to_check] = pool.map(check_filesystem_and_run, to_check)

#     # signal.pthread_sigmask(signal.SIG_UNBLOCK,[signal.SIGINT])

#     inpaint_ij = multiprocessing.Manager().list()
#     kw_pre=pre_process_kw
    
#     if pass_kw:
#         pre_process_kw0 = pre_process_kw

#     def loader(k):
#         nonlocal kw_pre # this nonlocal declaration shouldn't be necessary, but fixes a bug in some pyhton 3.10 versions

#         i,j = np.unravel_index(k, shape)
#         if results[i,j] is None:
#             path = savepaths[k]
#             file_exists = os.path.exists(path)
#             if file_exists:                
#                 with open(path, 'rb') as file:
#                     try:
#                         if pre_process is None:
#                             out = pickle.load(file)
#                         else:
#                             if pass_kw:
#                                 kw_pre={**pre_process_kw0, **kw[j]}

#                             out =  pre_process(pickle.load(file), **kw_pre)

#                         if verbose:
#                             print(f'loaded[{i}][{j}]: {path}  ({os.path.getsize(path)/(1024*1024)} mb)')
                        
#                         result =  out
#                     except:
#                         if verbose:
#                             print(f'exception loading/processing: {path}  ({os.path.getsize(path)/(1024*1024)} mb)')
#                         result = None

#             else:
#                 # print(f'inpainting[{i}][{j}]:')
#                 result = None

#             if result is None:
#                 inpaint_ij.append(k)


#             return result


  
      

#     if pre_process is None:
#         pool = ThreadPool(nodes=psutil.cpu_count()-1) #revert to simple if we just have threading for IO-bound stuff
        
#     needed = np.where(results_raveled==None)[0] 
#     results_raveled[needed] = pool.map(loader, needed) #multiprocessing for CPU-bound stuff


#     # results = np.reshape( results, shape)

#     if possibly_new_results and pre_process is not None and cache:
#         with open(cache, 'wb') as file:
#                 pickle.dump(results, file)
    
#     #inpaint after we have cached
#     if inpaint is not None:
#         for k in inpaint_ij:
#             results_raveled[k] = inpaint

#     types=set([type(r) for r in results_raveled])
#     is_scalar = [(not hasattr(r,'shape') ) or r.size==1 for r in results_raveled]
#     try:
#         if np.all(is_scalar):
#             results=np.array(results, dtype = first_item(types))
#     except:
#         pass
    
#     return results

def parameter_sweep_analyzer(params, simulations, analyzer, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None):
    results = parameter_sweep(params, simulations, savepaths=savepaths, overwrite=overwrite, pool=pool, pre_process=pre_process, inpaint=inpaint)

    analyzer(params, results)


def parameter_keywords_sweep_analyzer( params, simulations, analyzer, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None):
    results = sweep( params, simulations, savepaths=savepaths, overwrite=overwrite, pool=pool, pre_process=pre_process, inpaint=inpaint)

    analyzer(params, results)

if __name__ == '__main__':
    def dummy(force, visco=False, a=True, b=1.0):
        sig=signature_string()
        print(f'this is a dummy function, that was called with {sig}')

    dummy(1.3, b=1.2, a=False, visco=True)
    dummy(1.35)
