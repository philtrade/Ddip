import multiprocess as mp
from functools import partial
from typing import Callable
import os, inspect

__all__ = ['import_star', 'distributedly', 'mplaunch', 'ddplaunch']

def import_star(modules=[]):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    g = inspect.currentframe().f_back.f_globals
    new_imports = {}
    
    for mod in modules:
        m = __import__(mod, fromlist=['*'])
        for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
            new_imports[name] = getattr(m, name)
    g.update(new_imports)

def distributedly(fn):
    "A decorator that sets up environment variable necessary for distributed data parallel training."
    def run_in_ddp(*args, _i:int=None, _g:int=None, _ws:int=None, **kwargs):
        import torch
        assert _i is not None, ValueError("must provide rank in '_i'")
        assert _ws is not None, ValueError("must provide world size in '_ws'")
        os.environ["LOCAL_RANK"] = str(_i)
        os.environ["RANK"] = os.environ["LOCAL_RANK"] # Check "RANK" usage in fastai --> if _g is None else str(_g)
        os.environ["WORLD_SIZE" ] = str(_ws)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29500)
        os.environ["OMP_NUM_THREADS"] = str(1) # See https://github.com/pytorch/pytorch/pull/22501
        print(f"{os.getpid()} run_in_ddp LOCAL_RANK: {os.environ['LOCAL_RANK']}")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0))) 

        r = fn(*args, **kwargs)

        for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]:
            if k in os.environ: del os.environ[k]
        return r
    return run_in_ddp

def mplaunch(nprocs:int, fn:Callable, *args, rank0_parent:bool=True, host_rank:int=0, **kwargs):
    "A multiprocess function launcher that works in Jupyter Notebook as well"

    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    procs = []
    start = 1 if rank0_parent is True else 0
    ctx = mp.get_context("spawn")
      
    try:
        for rank in range(start, nprocs):
            kwargs.update({'_i':rank, '_g':rank+host_rank*nprocs, '_ws': nprocs})
            p = ctx.Process(target=fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()

        if rank0_parent: # use current process as rank-0 to execute the function
              kwargs.update({'_i':0, '_g':0 + host_rank*nprocs, '_ws': nprocs})
              r = fn(*args, **kwargs)
        else: r = procs
                            
        return r
    except Exception as e:
        raise Exception(e) from e
    finally:
        for p in procs: p.join()

def ddplaunch(nprocs, fn, *args, **kwargs):
    return mplaunch(nprocs, distributedly(fn), *args, **kwargs)
    