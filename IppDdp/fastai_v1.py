import sys, time, os, torch, atexit, dataclasses
from types import SimpleNamespace
from typing import List
from fastai.distributed import DistributedTrainer, DistributedRecorder
from fastai.torch_core import *
from fastai import basic_data, basic_train, core, text, train
from fastai.basic_train import Learner
from fastai.callbacks.lr_finder import LRFinder
import fastprogress
from fastprogress.fastprogress import master_bar, progress_bar, force_console_behavior, IN_NOTEBOOK

'''
FastAI specific setup
'''
FastaiSaver = SimpleNamespace(post_init = None, lr_find = None,
    Verbose = False, old_cbs = None, lr_finder_GPUs = {0})

def print_verbose(*args, **kwargs): FastaiSaver.Verbose and print(*args, **kwargs, flush=True)

def set_verbose(verbose:bool=True): FastaiSaver.Verbose = verbose

def lr_finder_ids(gpus:List[int]): #To-do: use set operations to allow removal.
    for i in gpus:
        if i < torch.cuda.device_count(): FastaiSaver.lr_find_GPUs.add(i)
    return FastaiSaver.lr_find_GPUs

def _post_init_DDP(learner):
    '''Make a freshly created Learner object to run in DDP mode when training.
    '''
    assert FastaiSaver.post_init, " Original fastai.Learner.__post_init__ not saved yet!"
    FastaiSaver.post_init(learner)
    learner.to_distributed(torch.cuda.current_device())

def ddpify_Learner_class():
    if FastaiSaver.post_init is None:  # Intercept Learner.__post_init__() to append our own handler
        FastaiSaver.post_init = Learner.__post_init__
        FastaiSaver.lr_find = Learner.lr_find
        Learner.__post_init__ = _post_init_DDP
        Learner.lr_find = lr_find_bypass

def restore_Learner_class():
    if FastaiSaver.post_init is not None:
        Learner.__post_init__ = FastaiSaver.post_init
        Learner.lr_find = FastaiSaver.lr_find
        FastaiSaver.post_init = FastaiSaver.lr_find = None

def to_non_distributed(learn:Learner):
    '''Undo the preparation for DDP mode on this learner instance.'''
    dist_cb = {DistributedTrainer, DistributedRecorder }
    for t in dist_cb:
        for cb in learn.callbacks:
            if isinstance(cb, t):
                print_verbose(f"Removing callback {type(cb)} from learner.")
                learn.callbacks.remove(cb)

def lr_find_bypass(learn:Learner, *args, **kwargs):
    assert FastaiSaver.lr_find, "Original lr_find() not saved yet.  Was _distrib_Learner(True) called?"
    to_non_distributed(learn)

    if rank_distrib() in FastaiSaver.lr_finder_GPUs:
        FastaiSaver.lr_find(learn, *args, **kwargs)
    else:
        LRFinder(learn).on_train_end()

    learn.to_distributed(torch.cuda.current_device())

def silent_console(silent:bool=True):
    "Turn off console progress bar output."
    fastprogress.fastprogress.NO_BAR = silent
    mbar, pbar = force_console_behavior()
    cls_list = [fastprogress.fastprogress, basic_train, basic_data,
        dataclasses.dataclass, text, text.data, core]
    for c in cls_list: c.master_bar, c.progress_bar = mbar,pbar

imports = '\n'.join([ 'import fastai, fastai.torch_core, torch, fastprogress',
    'from fastai.distributed import *', f"from {__name__} import initializer, finalizer, set_verbose, lr_find_bypass",])

def initializer():
    '''A few fastai_v1-specific housekeeping:
    0. Set defaults.device to the proper device, due to a bug in fastai.torch_core.py
        where the defaults.device is initialized without regard to the curren cuda device.
    1. Limit standard output to only the RANK 0 process.
    3. Intercept Learner constructor to call to_distributed() after Learner.__post_init__().
    '''
    import fastai.torch_core
    print_verbose(f"Entering fastai_init_ddp(): {os.getpid()}")
    if torch.cuda.is_available(): torch.cuda.set_device(torch.cuda.current_device()) # work-around for the fastai.torch_core.defaults.device maybe out of sync
    silence = rank_distrib() != 0
    silent_console(silence) # limit console output to only rank 0 GPU
    ddpify_Learner_class() # Learner class to call to
    return f"[Process {os.getpid()}] Rank {rank_distrib()} fastai initialized for distributed data parallel."

def finalizer():
    import fastai.torch_core
    restore_Learner_class() # Restore Learner post-initializer
    silent_console(False) # Let them sing again
