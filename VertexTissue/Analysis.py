import pickle, os, collections, concurrent, psutil, multiprocessing, time
# from jug import TaskGenerator
from collections.abc import Iterable
from pathos.pools import ProcessPool as Pool
from pathos.pools import ThreadPool as ThreadPool
import numpy as np
import networkx as nx


from .util import check_funcion_cache, get_creationtime, get_filenames, get_oldestfile, signature_string, signature_lists


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





def parameter_sweep(params, simulations, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None, cache=None):

    results = None
    shape=(len(params), len(simulations))

    if cache is not None and pre_process is not None:
        if cache == True:
            results, cache = check_funcion_cache(pre_process)


        
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

def parameter_keyword_sweep(params, func, kw={}, savepath_prefix='.', extension='.pickle', overwrite=False, pool=None, pre_process=None, inpaint=None):
   
    extension=extension.strip()
    if extension[0] != '.':
        extension='.'+extension

    if isinstance(kw, dict):
        kw=[kw]  

    results = None
    shape=(len(params), len(kw))

    if cache is not None and pre_process is not None:
        if cache == True:
            results, cache = check_funcion_cache(pre_process)


        
    if results is None or not (results.shape==shape):
        results= np.array([ [None]*len(simulations) ]*len(params))

    

    if pool is None:
        core_count = psutil.cpu_count(logical=False)
        pool = Pool(nodes=core_count-1)

    
    



    args, _ = signature_lists(func)



    shape=(len(params), len(kw))

    savepaths = multiprocessing.Manager().dict()

    def runner(ij):

        i,j=ij
        pars=params[i]
        if not isinstance(pars, Iterable):
            pars=(pars,)
        locals = {**{a:v for a,v in zip(args, pars)}, **kw[j]}
        path = os.path.join(savepath_prefix, signature_string(f=func, locals=locals) + extension)
        # savepaths_proxy_i = savepaths_proxy[i]
        # savepaths_proxy_i[j] = path
        savepaths[ij] = path
        missing_file = not os.path.exists(path)
        run = overwrite or (missing_file and inpaint is None)
        
        
        if run:
            result =  func(*pars, **kw[j])
        else:
            result = None


        return result

    results = np.reshape(pool.map(runner, np.ndindex(shape)), shape) #parallel for CPU-bound stuff

    def loader(ij):
       
        i,j=ij
        if results[i][j] is None:
            path = savepaths[ij]
            file_exists = os.path.exists(path)
            if file_exists: #and results[i][j] is None:
                
                
                with open(path, 'rb') as file:
                    try:
                        if pre_process is None:
                            out = pickle.load(file)
                        else:
                            out =  pre_process(pickle.load(file))

                        print(f'loaded[{i}][{j}]: {path}  ({os.path.getsize(path)/(1024*1024)} mb)')
                        
                        return out
                    except:
                        return inpaint

            else:
                # print(f'inpainting[{i}][{j}]:')
                return inpaint


  
    
    

    if pre_process is None:
        results = ThreadPool(nodes=psutil.cpu_count()-1).map(loader, np.ndindex(shape)) #threading for IO-bound stuff
    else:
        results = pool.map(loader, np.ndindex(shape))

    results = np.reshape( results, shape)


    return results

def parameter_sweep_analyzer(params, simulations, analyzer, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None):
    results = parameter_sweep(params, simulations, savepaths=savepaths, overwrite=overwrite, pool=pool, pre_process=pre_process, inpaint=inpaint)

    analyzer(params, results)


def parameter_keywords_sweep_analyzer( params, simulations, analyzer, savepaths=None, overwrite=False, pool=None, pre_process=None, inpaint=None):
    results = parameter_keyword_sweep( params, simulations, savepaths=savepaths, overwrite=overwrite, pool=pool, pre_process=pre_process, inpaint=inpaint)

    analyzer(params, results)

if __name__ == '__main__':
    def dummy(force, visco=False, a=True, b=1.0):
        sig=signature_string()
        print(f'this is a dummy function, that was called with {sig}')

    dummy(1.3, b=1.2, a=False, visco=True)
    dummy(1.35)
