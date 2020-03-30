import sys, time, os, torch
from types import SimpleNamespace
from typing import List
from fastprogress.fastprogress import force_console_behavior
from fastai2.torch_core import *  # progress_bar, master_bar
# from fastai2.notebook.core import *  # IN_NOTEBOOK
from fastai2.distributed import *
from fastai2.learner import Learner

import fastprogress

FastaiSaver = SimpleNamespace(post_init = None,)
Config = SimpleNamespace(Verbose = False, pid = os.getpid())

imports = [ 'import fastai2, torch, fastprogress, fastai2.torch_core',
    # 'fastai2.text.learner, fastai2.text.core, fastai2.callback.progress, fastai2.data.external',
    'from fastai2.distributed import *',
    f"from {__name__} import initializer, finalizer, set_verbose, silent_console",]

def set_verbose(verbose:bool=True): Config.Verbose = verbose
def print_verbose(*args, **kwargs): Config.Verbose and print(f"Proc [{Config.pid}]", *args, **kwargs, flush=True)

'''
def _post_init_DDP(learner):
    "Make a freshly created Learner object to run in DDP mode when training."
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
'''

def silent_console(silent:bool=True):
    "Turn off console progress bar output."
    import fastai2.torch_core, fastai2.text.learner, fastai2.text.core, fastai2.callback.progress, fastai2.data.external
    fastprogress.fastprogress.NO_BAR = silent
    mbar, pbar = force_console_behavior()
    cls_list = [fastprogress.fastprogress, fastai2.torch_core, fastai2.text.learner,
        fastai2.text.core, fastai2.callback.progress, fastai2.data.external,
        ]
    for c in cls_list: c.master_bar, c.progress_bar = mbar,pbar

def initializer():
    '''A few fastai_v1-specific housekeeping:
    1. Fix defaults.device out of sync bug in fastai.torch_core.py
    2. Limit progress bar standard output to only RANK 0 process.
    3. Intercepting Learner.__post_init__(), call to_distributed() at the end.
    '''
    import fastai2.torch_core
    import inspect
    print_verbose(f"Entering {__name__}.{inspect.currentframe().f_code.co_name}(): {os.getpid()}")
    # work-around for the fastai.torch_core.defaults.device maybe out of sync
    if torch.cuda.is_available(): torch.cuda.set_device(torch.cuda.current_device())
    silence = rank_distrib() != 0
    silent_console(silence) # limit console output to only rank 0 GPU
    # ddpify_Learner_class() # Patch the Learner class to route/bypass some methods
    return f"[Process {os.getpid()}] Rank {rank_distrib()} fastai initialized for distributed data parallel."

def finalizer():
    import fastai2.torch_core
    # restore_Learner_class() # Restore Learner post-initializer
    silent_console(False) # Let them sing again
