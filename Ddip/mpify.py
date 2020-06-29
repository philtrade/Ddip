import os, inspect, multiprocess as mp
from typing import Callable
from contextlib import AbstractContextManager

__all__ = ['import_star', 'ranch', 'TorchDistribContext',]

def import_star(modules=[]):
    "Apply `from module import '*'` into caller's frame from a list of modules."
    cf = inspect.currentframe()
    g = cf.f_back.f_globals
    try:
        for mod in modules:
            try:
                m = __import__(mod, fromlist=['*'])
                to_import = {}
                for name in getattr(m, "__all__", [n for n in dir(m) if not n.startswith('_')]):
                    to_import[name] = getattr(m, name)
                g.update(to_import)
            except Exception as e: raise ImportError(f"Failed to import module {mod}") from e
    finally:
        del cf   # Recommendation from https://docs.python.org/3/library/inspect.html#the-interpreter-stack

def _contextualize(fn:Callable, cm:AbstractContextManager):
    "Wraps a function within a context manager.."
    def _cfn(*args, **kwargs):
        with cm: return fn(*args, **kwargs)
    return _cfn

def ranch(nprocs:int, fn:Callable, *args, parent_rank:int=0, host_rank:int=0, ctx=None, **kwargs):
    "Launch a function among a group of ranked processes.  Parent process can participate.  Works in interactive IPython/Jupyter notebook"
    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    children_ranks = list(range(nprocs))
    if parent_rank is not None:
        assert 0 <= parent_rank < nprocs, ValueError(f"Out of range parent_rank:{parent_rank}, must be 0 <= parent_rank < {nprocs}")
        children_ranks.pop(parent_rank)

    multiproc_ctx = mp.get_context("spawn")
    procs = []
    try:
        os.environ["WORLD_SIZE" ], base_rank = str(nprocs), host_rank * nprocs
        if ctx is not None: fn = _contextualize(fn, ctx)
        for rank in children_ranks:
            os.environ.update({"LOCAL_RANK":str(rank), "RANK":str(rank + base_rank)})
            p = multiproc_ctx.Process(target=fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()

        # execute the function in current process as rank-{parent_rank}
        if parent_rank is not None:
            os.environ.update({"LOCAL_RANK":str(parent_rank), "RANK":str(parent_rank + base_rank)})
            return fn(*args, **kwargs)
        else: return procs
    except Exception as e: raise Exception(e) from e
    finally:
        for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]: os.environ.pop(k, None)
        for p in procs: p.join()

import torch
class TorchDistribContext(AbstractContextManager):
    "A context manager to customize setup/teardown of pytorch distributed data parallel environment."
    def __init__(self, *args, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        self._a, self._p, self._nt = addr, port, num_threads

    def __enter__(self):
        os.environ.update({"MASTER_ADDR":self._a, "MASTER_PORT":str(self._p),
                           "OMP_NUM_THREADS":str(self._nt)})
        if torch.cuda.is_available(): torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        # should torch.cuda.initialize_distributed_group() go here?
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for k in ["MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]: os.environ.pop(k, None)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return exc_type is None
