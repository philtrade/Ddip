import sys, time, os, torch, atexit, dataclasses
from types import SimpleNamespace
from typing import List
from fastai.torch_core import *
from fastai import basic_data, basic_train, core, text, train
from fastai.basic_train import Learner
from fastai.callbacks.lr_finder import LRFinder

import fastprogress
from fastprogress.fastprogress import master_bar, progress_bar, force_console_behavior, IN_NOTEBOOK

FastaiSaver = SimpleNamespace(post_init = None,)
Config = SimpleNamespace(Verbose = False, pid = os.getpid())

imports = [ 'import fastai, fastai.torch_core, torch, fastprogress', 'from fastai.distributed import *',
    f"from {__name__} import initializer, finalizer, set_verbose",]

def set_verbose(verbose:bool=True): Config.Verbose = verbose
def print_verbose(*args, **kwargs): Config.Verbose and print(f"Proc [{Config.pid}]", *args, **kwargs, flush=True)

def _post_init_DDP(learner):
    '''Make a freshly created Learner object to run in DDP mode when training.'''
    assert FastaiSaver.post_init, " Original fastai.Learner.__post_init__ not saved yet!"
    FastaiSaver.post_init(learner)
    learner.to_distributed(torch.cuda.current_device())

def ddpify_Learner_class():
    if FastaiSaver.post_init is None:  # Intercept Learner.__post_init__() to append our own handler
        FastaiSaver.post_init = Learner.__post_init__
        Learner.__post_init__ = _post_init_DDP

def restore_Learner_class():
    if FastaiSaver.post_init is not None:
        Learner.__post_init__ = FastaiSaver.post_init
    FastaiSaver.post_init = None

def silent_console(silent:bool=True):
    "Turn off console progress bar output."
    fastprogress.fastprogress.NO_BAR = silent
    mbar, pbar = force_console_behavior()
    cls_list = [fastprogress.fastprogress, basic_train, basic_data,
        dataclasses.dataclass, text, text.data, core]
    for c in cls_list: c.master_bar, c.progress_bar = mbar,pbar

def initializer():
    '''A few fastai_v1-specific housekeeping:
    1. Fix defaults.device out of sync bug in fastai.torch_core.py
    2. Limit progress bar standard output to only RANK 0 process.
    3. Intercepting Learner.__post_init__(), call to_distributed() at the end.
    '''
    import fastai.torch_core
    print_verbose(f"Entering fastai_init_ddp(): {os.getpid()}")
    # work-around for the fastai.torch_core.defaults.device maybe out of sync
    if torch.cuda.is_available(): torch.cuda.set_device(torch.cuda.current_device())
    silence = rank_distrib() != 0
    silent_console(silence) # limit console output to only rank 0 GPU
    ddpify_Learner_class() # Patch the Learner class to route/bypass some methods
    return f"[Process {os.getpid()}] Rank {rank_distrib()} fastai initialized for distributed data parallel."

def finalizer():
    import fastai.torch_core
    restore_Learner_class() # Restore Learner post-initializer
    silent_console(False) # Let them sing again
