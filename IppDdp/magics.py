import sys, time, os, re, ipyparallel, torch, atexit, IPython, dataclasses
from collections import OrderedDict
from typing import List
from types import SimpleNamespace
from torch.distributed import *
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython.core.display import clear_output
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from ipyparallel import AsyncResult
from .torchDDP import Ddp

this = sys.modules[__name__]
this.defaults = { 'appname' : 'fastai-v1',}

Config = SimpleNamespace(Verbose = True)

@magics_class
class IppDdp(Magics):
    '''IppDdp is an ipython extension of line and cell magics to harness the pytorch
    distributed data parallel (DDP) execution over the ipyparallel cluster.
    '''
    def __init__(self, shell:IPython.InteractiveShell, **kwargs):
        super(IppDdp, self).__init__(shell=shell) # will setup self.shell
        self.init_ddp(**kwargs)

    def init_ddp(self, **kwargs):
        '''Initialize a new DDP group, reset states of ipython magic, and associated applicaiton to None.'''
        self._autoddpx = None # Flag to control if parallel execution is by default ON or OFF
        self.ddp = Ddp(**kwargs) # Controller for DDP, and the underlying ipyparallel cluster

    def __del__(self):
        if self.ddp: self.ddp.shutdown_cluster()

    def gpu_str2list(self, g_str:List[str]):
        if 'all' in g_str:
            gpus = list(range(torch.cuda.device_count()))
        else:
            gpus = list(OrderedDict.fromkeys([ int(g) for g in re.split(r',\s*|\s+', ','.join(g_str))]))
        return gpus

    def prepender(self, lines:List[str]):
        '''Prepend automatic DDP execution cell magic, unless the cell begins with '%', i.e. an explicit magic. https://github.com/jdanbrown/potoo/blob/master/potoo/default_magic_magic.py'''
        if self._autoddpx and lines and (not lines[0].startswith('%')):
            lines.insert(0, f"{self._autoddpx}\n")
        return lines
    
    @line_magic
    def nop(self, line:str):
        '''Pure No-Op line object, used as a marker to temporarily disable the `prepender()` from inserting anything. '''
        pass

    @line_magic
    def autoddpx(self, line:str):
        '''Prepend %%ddpx to subsequent cells so that they will run on the distributed data parallel cluster.'''
        '''Todo: free memory after each run'''
        if line:
            hooks = self.shell.input_transformers_cleanup
            args = line.split(None)
            if args[0] == "off": # Unregister the prepender
                self._autoddpx = None
                while self.prepender in hooks: hooks.remove(self.prepender)
            else: # Register the prepender
                if args[0] == "on": args.pop(0)
                self._autoddpx = "%%ddpx " + ' '.join(args)
                if self.prepender not in hooks: hooks.append(self.prepender)
        return f"Automatic DDP execution: {self._autoddpx} on GPUs {self.ddp.ddp_group}"

    @line_magic
    def ddpstop(self, line=''):
        if self.ddp:
            Config.Verbose and print("%ddpstop shutting down cluster....", flush=True, file=sys.stderr)
            self.ddp.shutdown_cluster()
            self.ddp = None

    @magic_arguments()
    @argument('-g', '--gpus', dest='gpus', type=str, nargs='+', help="list of GPU ids, or 'all' to specify all GPUs available.")
    @argument('-a', '--app', dest='appname', default=this.defaults['appname'], type=str)
    @argument('-r', '--restart', dest='restart', action="store_const", const=True, help="Restart all engine processes.")
    @argument('-v', '--verbose', dest='verbose', nargs='?', type=str, const="True", choices=["True", "False"],help='print a message at each execution.')
    @line_magic
    def ddprep(self, line=''):
        '''%ddprep -- line magic to setup/tear down the cluster as a DDP training group, app-specific handling of object'''
        '''Todo: add -i for info on current config'''
        if self.shell is None: raise RuntimeError("%%ddpx: Not in an ipython Interactive shell!")
        args = parse_argstring(self.ddprep, line)

        if args.verbose: Config.Verbose = args.verbose == "True"
        if args.restart: self.ddpstop()
        if not self.ddp: self.init_ddp()
        self.ddp.set_verbose(Config.Verbose)

        if args.gpus:
            gpus = self.gpu_str2list(args.gpus)

            if self.ddp.ddp_group == gpus: # Group desired is already there?
                print(f"%ddprep DDP group unchanged, GPUs ids: {gpus}")
                return  # same group or empty group => do nothing
            
            self.ddp.exit_group() # Exit old DDP group if exists
                
            self.ddp.new_group(gpus=gpus, appname=args.appname)
            self.shell.run_line_magic("pxconfig", "--verbose" if Config.Verbose else "--no-verbose")        

    @magic_arguments()
    @argument('-q', '--quiet', dest='quiet', action='store_true', default=False, help="Display any stdout only after task is finished, skip all the transient, real-time output.")
    @argument('-f', '--free', dest='gc', action='store_const', const=True, default=True,  help="Free up memory on each engine at the completion of the cell")
    @argument('-g', '--gpus', dest='gpus', type=str, nargs='+', help="list of GPU ids, or 'all' to specify all GPUs available.")
    @argument('--local', '-L', dest='local', nargs='?', type=str, choices=["too", "only"], const='too',
        help="Run the cell locally in this iPython, either before running on the cluster, or 'only' locally and don't run on cluster at all.  Default to `too`")
    @cell_magic
    def ddpx(self, line, cell):
        '''%%ddpx - Parallel execution on cluster, allows transient output be displayed.'''
        if self.shell is None: raise RuntimeError("%%ddpx: Not in an ipython Interactive shell!")
        assert self.ddp and self.ddp.ddp_group, "%%ddpx: DDP group does not exist yet.  Have you run %ddprep -g <gpu list>?"
        run_on = { 'too' : 'local ipython and ', 'only' : 'local ipython ONLY, SKIPPING '}

        args = parse_argstring(self.ddpx, line)

        # where to run the cell
        gpus = self.gpu_str2list(args.gpus) if args.gpus else self.ddp.ddp_group
        where = f"{run_on.get(args.local,'')}" + f"cluster (GPUs: {gpus})"
        if Config.Verbose: print(f"%%ddpx {line}: Running cell on {where}", flush=True)

        # If --local, run in local ipython process/namespace first, then on the specified GPUs.
        if args.local: self.shell.run_cell("%nop\n"+cell, silent = args.quiet)
        if args.local == 'only': return

        self.ddp.run_cell(cell, gpus=gpus, quiet=args.quiet, gc=args.gc)
        
    @line_magic
    def ddpobj(self, line:str=''):
        if line: self.shell.push({line.split(None,1)[0] : self.ddp})
        else: print("%ddpobj requires one argument: a variable name to store the DDP object.", file=sys.stderr)

def unload_ipython_extension(ipython):
    ipython.magics_manager.registry.pop('IppDdp')

def load_ipython_extension(ipython):
    ipython.register_magics(IppDdp)
