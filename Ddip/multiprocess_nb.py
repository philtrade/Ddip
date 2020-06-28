import multiprocess as mp
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Callable
import os, inspect

__all__ = ['import_star', 'mplaunch', 'ddplaunch', 'mpify', 'ddplaunch','TorchDistribContext']

def import_star(modules=[]):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    g = inspect.currentframe().f_back.f_globals
    new_imports = {}
    
    for mod in modules:
        m = __import__(mod, fromlist=['*'])
        for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
            new_imports[name] = getattr(m, name)
    g.update(new_imports)

def mplaunch(nprocs:int, fn:Callable, *args, rank0_parent:bool=True, host_rank:int=0, **kwargs):
    "A multiprocess function launcher that works in interactive IPython/Jupyter notebook"
    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    procs = []
    start = 1 if rank0_parent is True else 0
    ctx = mp.get_context("spawn")

    try:
        os.environ["WORLD_SIZE" ], base_rank = str(nprocs), host_rank * nprocs
        for rank in range(start, nprocs):
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["RANK"] = str(rank + base_rank)
            p = ctx.Process(target=fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()

        # execute the function in current process as rank-0 
        os.environ["LOCAL_RANK"], os.environ["RANK"]= str(0), str(0 + base_rank)
        return fn(*args, **kwargs) if rank0_parent else procs

    except Exception as e:
        raise Exception(e) from e
    finally:
        for p in procs: p.join()

def mpify(fn):
    "Decorator to convert a function to run by mplaunch, with a pre-setup, post-cleanup routine."
    def _mpfunc_with_rank(*args, rank_ctx:Callable=nullcontext(), **kwargs):
        with rank_ctx: return fn(*args, **kwargs)
    return _mpfunc_with_rank

class TorchDistribContext():
    "A context manager to setup/teardown the distributed data parallel in pytorch."
    all_keys = ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]
    def __init__(self, *args, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        self.addr = os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["OMP_NUM_THREADS"] = str(num_threads) # See https://github.com/pytorch/pytorch/pull/22501

    def __enter__(self):
        import torch
        print(f"{self.__class__}.__enter__(): {[os.environ[k] for k in self.all_keys]}")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        # should torch.cuda.initialize_distributed_group() go here?
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for k in self.all_keys:
            if k in os.environ: del os.environ[k]
        return exc_type is None

def ddplaunch(nprocs, fn, *args, rank_ctx=None, **kwargs):
    if rank_ctx is None: rank_ctx = TorchDistribContext()
    "Convenience multiproc launcher for torch distributed data parallel setup, accept customized TorchDistribContext object."
    return mplaunch(nprocs, mpify(fn), *args, rank_ctx=rank_ctx, **kwargs)

    