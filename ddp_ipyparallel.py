import sys, time, os, re, ipyparallel, torch, atexit, IPython, dataclasses, subprocess, signal
from collections import OrderedDict
from typing import List
from torch.distributed import *

Debug = True
Verbose = True

def _debug(*args, **kwargs):
    if Debug: print(*args, file=sys.stderr, **kwargs)

def join_group_single(g_rank:int, l_rank:int, gpu:int, ws:int):
    import os, torch
    os.environ["RANK"] = str(g_rank) # Global rank
    os.environ["LOCAL_RANK"] = str(l_rank) # Local rank
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500)
    os.environ["WORLD_SIZE"] = str(ws)
    os.environ["OMP_NUM_THREADS"] = str(1) # See https://github.com/pytorch/pytorch/pull/22501
    torch.cuda.set_device(gpu)
    if ws > 1: torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return os.environ["LOCAL_RANK"]

def exit_group_single():
    for i in ["RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "OMP_NUM_THREADS"]: os.environ.pop(i)
    torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


class IppCluster():
    """Start/stop of an ipyparallel cluster aka 'ipcluster, and access to cluster engines."""
    cid = "ippdpp_c"
    def __init__(self, n:int=0, engine_wait:float=15.0, **kwargs):
        popen_cmd = ["ipcluster", "start", f"--cluster-id={IppCluster.cid}", "--daemonize"]
        if n > 0: popen_cmd.append(f"--n={n}")

        cluster_proc = subprocess.Popen(popen_cmd) # Ignore stdout and stderr from ipcluster
        try: cluster_proc.wait(10)
        except subprocess.TimeoutExpired as e: raise TimeoutError("Timed out {popen_cmd}: {e}")

        try: self.client = ipyparallel.Client(timeout=engine_wait,cluster_id=f"{IppCluster.cid}")
        except (IOError,TimeoutError) as e: raise Exception(f"ipyparallel Client() failed: {e}")
        print(f"Connecting to ipyparallel cluster.", file=sys.stderr, end='', flush=True)
        while engine_wait > 0:
            try:
                self.px_view = self.client[:] # This will turn on ipyparallel's ipython line and cell magics
                engine_wait = 0
                pm = self.px_view.apply_async(os.getpid).get_dict()
                self.e_pids = [ pm[k] for k in sorted(pm) ]
                e_ppid = self.client[0].apply_sync(os.getppid)
                self.e_ppid = e_ppid

                def carefree_kill():
                    '''cleanup routine not tied to the object itself, ensure the object can be garbage collected after 'del''''
                    subprocess.call(["ipcluster", "stop", f"--cluster-id={IppCluster.cid}"])
                    try:
                        os.kill(e_ppid, signal.SIGINT)
                        Verbose and print("pausing 3 seconds to let cluster shut down....", flush=True, file=sys.stderr)
                        time.sleep(3.0)
                    except ProcessLookupError: pass
                atexit.register(carefree_kill)
                self.carefree_kill = carefree_kill

            except ipyparallel.error.NoEnginesRegistered as e:
                print('.', file=sys.stderr, end='', flush=True)
                time.sleep(1)
                engine_wait -= 1

    def __del__(self): self.shutdown()

    def interrupt_engines(self, eids:List[int]):
        '''Send SIGINT to a list of ipyparallel engines'''
        for i in eids: i < len(self.e_pids) and os.kill(self.e_pids[i], signal.SIGINT)

    def shutdown(self):
        '''Shutdown ipyparallel cluster -- this will kill all processes started by the 'ipcluster start' command.'''
        self.carefree_kill()
        self.px_view = self.client = self.e_pids = self.e_ppid = None
        
class Ddp():
    def __init__(self, **kwargs):
        assert torch.cuda.is_available(), "CUDA not available! (Try reloading cuda driver?)"
        n_engines=torch.cuda.device_count()
        self.cluster = IppCluster(n=n_engines, **kwargs)
        self.ddp_group = None

    def __del__(self): self.shutdown_cluster()

    def new_group(self, gpus:List[int], node_rank=0, world_size=0):
        '''Configure a torch distributed process group of GPUs over a ipyparallel cluster.
        Returns a list of GPU ids of the group formed.'''
        n_gpu = len(gpus)
        assert n_gpu <= len(self.cluster.client), f"More GPU ({gpus}) than ipyparallel engines ({len(self.cluster.client)})"
        assert max(gpus) < torch.cuda.device_count(), f"Invalid GPU id {max(gpus)}, highest allowed is {torch.cuda.device_count()-1}"

        Verbose and print(f"Initializing torch distributed group with GPUs {gpus}", flush=True)

        if world_size==0: world_size = n_gpu
        for rank,gpu in enumerate(gpus):
            dist_rank = n_gpu * node_rank + rank # see torch/distributed/launch.py
            self.cluster.client[gpu].push(dict(g_rank=dist_rank, l_rank=rank, gpu=gpu, ws=world_size))

        # ipyparallel client[] accepts list of ints as slice indices.
        self.cluster.px_view = self.cluster.client[gpus]
        self.cluster.px_view.execute('from ippddp.ddp_ipyparallel import join_group_single, exit_group_single')
        self.cluster.px_view.execute('r = join_group_single(g_rank=g_rank, l_rank=l_rank, gpu=gpu, ws=ws)', block=True)
        print("Local Ranks initialized: ", [ f"GPU{k}={v}" for k, v in self.cluster.px_view.pull('r').get_dict().items()], flush=True)
        self.ddp_group = gpus

    def gpus_str(self):
        return ','.join(map(str, self.ddp_group))

    def exit_group(self):
        if self.ddp_group is None: return
        Verbose and print(f"DDP.exit_group(): {self.ddp_group}", flush=True)
        self.cluster.px_view.execute('exit_group_single()',block=True)
        self.cluster.px_view = self.ddp_group = None

    def shutdown_cluster(self):
        self.exit_group()
        if self.cluster:
            self.cluster.shutdown()
            self.cluster = None
