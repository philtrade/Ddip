import multiprocess as mp
from typing import Callable
import os, inspect

__all__ = ['starimport', 'distributedly', 'mplaunch', 'ddplaunch']

def starimport(modules=[]):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    g = inspect.currentframe().f_back.f_globals
    new_imports = {}
    
    for mod in modules:
        m = __import__(mod, fromlist=['*'])
        for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
            new_imports[name] = getattr(m, name)
    g.update(new_imports)

def distributedly(fn, rank:int=None):
    "A decorator that sets up environment variable necessary for distributed data parallel training."
    if rank is not None: os.environ["LOCAL_RANK"] = str(rank)
    def run_in_ddp(*args, **kwargs):
        import torch
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29500)
        os.environ["OMP_NUM_THREADS"] = str(1) # See https://github.com/pytorch/pytorch/pull/22501
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0))) 

        r = fn(*args, **kwargs)

        for k in ["MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]:
            if k in os.environ: del os.environ[k]
        return r
    return run_in_ddp

def mplaunch(nprocs:int, fn:Callable, *args, rank0_parent:bool=True, decorator=None, **kwargs):
    "A multiprocess function launcher that works in Jupyter Notebook as well"

    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    procs = []
    start = 1 if rank0_parent is True else 0
    ctx = mp.get_context("spawn")
                              
    def set_rank(rank): os.environ["LOCAL_RANK"] = os.environ['RANK'] = str(rank)
      
    try:
        os.environ["WORLD_SIZE"] = str(nprocs)
        for rank in range(start, nprocs):
            set_rank(rank)
            if decorator: fn = decorator(fn, rank)
            p = ctx.Process(target=fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()

        if rank0_parent:
            set_rank(0)
            r = fn(*args, **kwargs)
        else: r = procs
                            
        return r
    except Exception as e:
        raise Exception(e) from e
    finally:
        del os.environ["WORLD_SIZE"]
        for p in procs: p.join()

def ddplaunch(nprocs, fn, *args, **kwargs):
    return mplaunch(nprocs, fn, *args, decorator=distributedly, **kwargs)
    