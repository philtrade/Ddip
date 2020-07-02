import os, inspect, multiprocess as mp
from typing import Callable
from contextlib import AbstractContextManager
import torch

__all__ = ['import_star', 'ranch', 'TorchDDPCtx', 'in_torchddp']

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
    "Wrap a context manager around a function's execution."
    def _cfn(*args, **kwargs):
        with cm: return fn(*args, **kwargs)
    return _cfn

def ranch(nprocs:int, fn:Callable, *args, parent_rank:int=0, host_rank:int=0, ctx=None, **kwargs):
    '''Launch `fn(*args, **kwargs)` to `nprocs` spawned processes. Local rank, global rank (multiple hosts),
       and world size are set in os.environ['LOCAL_RANK','RANK','WORLD_SIZE'] respectively.
       Parent process can participate as rank_{parent_rank}.
       Can optionally apply a context manager `ctx` around `fn(...)`.'''
    assert nprocs > 0, ValueError("nprocs: # of processes to launch must be > 0")
    children_ranks = list(range(nprocs))
    if parent_rank is not None:
        assert 0 <= parent_rank < nprocs, ValueError(f"Out of range parent_rank:{parent_rank}, must be 0 <= parent_rank < {nprocs}")
        children_ranks.pop(parent_rank)

    multiproc_ctx = mp.get_context("forkserver")

    procs = []
    try:
        os.environ["WORLD_SIZE" ], base_rank = str(nprocs), host_rank * nprocs
        target_fn = _contextualize(fn, ctx) if ctx else fn

        for rank in children_ranks:
            os.environ.update({"LOCAL_RANK":str(rank), "RANK":str(rank + base_rank)})
            p = multiproc_ctx.Process(target=target_fn, args=args, kwargs=kwargs)
            procs.append(p)
            p.start()

        if parent_rank is not None: # also run it in current process at a rank
            os.environ.update({"LOCAL_RANK":str(parent_rank), "RANK":str(parent_rank + base_rank)})
            return target_fn(*args, **kwargs)
        else: return procs
    except Exception as e: raise Exception(e) from e
    finally:
        for k in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]: os.environ.pop(k, None)
        for p in procs: p.join()

class TorchDDPCtx(AbstractContextManager):
    "Setup/teardown Torch DDP when entering/exiting a `with` clause."
    def __init__(self, *args, addr:str="127.0.0.1", port:int=29500, num_threads:int=1, **kwargs):
        self._a, self._p, self._nt = addr, port, num_threads
        self._myddp, self._backend = False, 'gloo' # default to CPU backend

    def __enter__(self):
        os.environ.update({"MASTER_ADDR":self._a, "MASTER_PORT":str(self._p),
                           "OMP_NUM_THREADS":str(self._nt)})
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
            self._backend = 'nccl'
        if not torch.distributed.is_initialized():
            print(f"rank[{os.environ['LOCAL_RANK']}] proc {os.getpid()} Initializing torch DDP: world size: {os.environ['WORLD_SIZE']}, backend: {self._backend}", flush=True)
            torch.distributed.init_process_group(backend=self._backend, init_method='env://')
            self._myddp = torch.distributed.is_initialized()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._myddp: torch.distributed.destroy_process_group()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        for k in ["MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]: os.environ.pop(k, None)
        return exc_type is None

def in_torchddp(nprocs:int, fn:Callable, *args, ctx:TorchDDPCtx=None, **kwargs):
    "Launch `fn(*args, **kwargs)` in Torch DDP group of `nprocs` processes.  Can customize the TorchddpCtx context."
    return ranch(nprocs, fn, *args, ctx = TorchDDPCtx() if ctx is None else ctx, **kwargs)