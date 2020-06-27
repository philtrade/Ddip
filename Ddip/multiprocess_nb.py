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
    '''
    A multiprocess function launcher that works in IPython/Jupyter notebook.
    Local rank, host rank, and world size are passed in as keyword arguments: `_i, _g, and _ws`.
    '''
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
    
class _null_ctx(nullcontext): # "Null context whose constructor ignores input parameters"
    def __init__(self, *args, **kwargs): super().__init__()
    
def mpify(fn):
    "Decorator to convert a function to run by mplaunch, with a pre-setup, post-cleanup routine."
    def _mpfunc_with_rank(*args, _i:int=None, _g:int=None, _ws:int=None, rank_ctx:Callable=_null_ctx, **kwargs):
        with rank_ctx(local_rank=_i, global_rank=_g, world_size=_ws) as c:
            r = fn(*args, **kwargs)
            return r
    return _mpfunc_with_rank

class TorchDistribContext():
    "A context manager to setup/teardown the distributed data parallel in pytorch."
    
    def __init__(self, *args, addr:str="127.0.0.1", port:int=29500, num_threads=1, **kwargs):
        "Set up group-wide communication.  Per process rank info can be customized in __call__()"
        self._addr, self._port, self._num_threads = addr, port, num_threads
        self._i, self._g, self._ws = None, None, None

    def __call__(self, local_rank:int, global_rank:int, world_size:int, *args, **kwargs):
        "Allow runtime dynamic setting of ranks and world size in a `with`-statement."
        self._i, self._g, self._ws = local_rank, global_rank, world_size
        return self
        
    def __enter__(self):
        import torch
        os.environ["LOCAL_RANK"] = str(self._i)
        os.environ["RANK"] = str(self._g)
        os.environ["WORLD_SIZE" ] = str(self._ws)
        os.environ["MASTER_ADDR"] = self._addr
        os.environ["MASTER_PORT"] = str(self._port)
        os.environ["OMP_NUM_THREADS"] = str(self._num_threads) # See https://github.com/pytorch/pytorch/pull/22501
        print(f"Entering: local_rank {self._i}, rank {self._g}, ws {self._ws}, addr {self._addr}, port {self._port}, num_threads {self._num_threads}")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        # should torch.cuda.initialize_distributed_group() go here?
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for k in ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS"]:
            if k in os.environ: del os.environ[k]
        return exc_type is None

def ddplaunch(nprocs, fn, *args, rank_ctx=TorchDistribContext(), **kwargs):
    return mplaunch(nprocs, mpify(fn), *args, rank_ctx=rank_ctx, **kwargs)

    