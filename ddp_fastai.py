import sys, time, os, torch, atexit, dataclasses
from fastai.distributed import *
from fastai.torch_core import *
from fastai import basic_data, basic_train, core, text
from fastai.basic_train import Learner
import fastprogress
from fastprogress.fastprogress import master_bar, progress_bar, force_console_behavior, IN_NOTEBOOK

'''
FastAI specific setup
'''
class FastaiDDP():
    _original_Learner_pi_:callable = None

    @staticmethod
    def _distributed_Learner_post_init(learner):
        FastaiDDP._original_Learner_pi_(learner)
        learner.to_distributed(torch.cuda.current_device())

    @staticmethod
    def _distrib_Learner(switch_on:bool):
        if switch_on:
            if FastaiDDP._original_Learner_pi_ is None:  # Intercept Learner.__post_init__() to append our own handler
                FastaiDDP._original_Learner_pi_ = Learner.__post_init__
                Learner.__post_init__ = FastaiDDP._distributed_Learner_post_init
        else:
            if FastaiDDP._original_Learner_pi_ is not None:
                Learner.__post_init__ = FastaiDDP._original_Learner_pi_

    @staticmethod    
    def silent_console(silent:bool=True):
        "Turn off console progress bar output."
        fastprogress.fastprogress.NO_BAR = silent
        mbar, pbar = force_console_behavior()
        cls_list = [fastprogress.fastprogress, basic_train, basic_data,
            dataclasses.dataclass, text, text.data, core]
        for c in cls_list: c.master_bar, c.progress_bar = mbar,pbar

    @staticmethod
    def fastai_init_ddp():
        '''Do a few housekeeping:
        0. Set defaults.device to the proper device, due to a bug in fastai.torch_core.py
           where the defaults.device is initialized without regard to the curren cuda device.
        1. Limit standard output to only the RANK 0 process.
        3. Intercept Learner constructor to call to_distributed() after Learner.__post_init__().
        '''
        print(f"Entering fastai_init_ddp(): {os.getpid()}", flush=True)
        if torch.cuda.is_available(): torch.cuda.set_device(torch.cuda.current_device()) # work-around for the fastai.torch_core.defaults.device maybe out of sync
        silence = rank_distrib() != 0
        FastaiDDP.silent_console(silence) # limit console output to only rank 0 GPU
        FastaiDDP._distrib_Learner(switch_on=True) # Update Learner post-initializer

    @staticmethod
    def fastai_finalize_ddp():
        import fastai.torch_core
        FastaiDDP._distrib_Learner(switch_on=False) # Restore Learner post-initializer
        FastaiDDP.silent_console(False) # Let them sing again
        fastai.torch_core.defaults.device = 'cpu'
