import sys, time, os, re, ipyparallel, torch, atexit, IPython, dataclasses
from collections import OrderedDict
from typing import List
from types import SimpleNamespace
from torch.distributed import *
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic, line_cell_magic
from IPython.core.display import clear_output
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from ipyparallel import AsyncResult
from .torchDDP import Ddp

Config = SimpleNamespace(Verbose = True, DefaultApp = 'fastai_v1')

@magics_class
class DdpMagic(Magics):
    '''An iPython extension to harness the pytorch distributed data parallel (DDP)
       execution over the ipyparallel cluster.
    '''
    _no_mod = "#DdpMagic_NO_MOD"

    def __init__(self, shell:IPython.InteractiveShell, **kwargs):
        super(DdpMagic, self).__init__(shell=shell) # will setup self.shell
        self.init_ddp(**kwargs)

    def init_ddp(self, **kwargs):
        '''Initialize a new DDP group, reset states of ipython magic, and associated applicaiton to None.'''
        self._autodip = None # Flag to control if parallel execution is by default ON or OFF
        self.ddp = Ddp(**kwargs) # Controller for DDP, and the underlying ipyparallel cluster

    def info(self):
        r = [ self.ddp.info(), f"Automatic DDP execution: {self._autodip or 'Off'}" ]
        return '\n'.join(r)

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
        if self._autodip and lines and (not lines[0].startswith('%')) and (not lines[0].startswith(DdpMagic._no_mod)):
            lines.insert(0, f"{self._autodip}\n")
        return lines

    @magic_arguments()

    @line_magic
    @argument('OnOff', type=str, choices=["on", "off"], nargs='?', help="Turn on auto-%%dip for the cells after this one.")
    def autodip(self, line:str):
        '''Prepend %%dip to subsequent cells so that they will run on the distributed data parallel cluster.'''
        '''Todo: free memory after each run'''
        args = parse_argstring(self.autodip, line)
        hooks = self.shell.input_transformers_cleanup

        if args.OnOff == "off": # Unregister the prepender
            self._autodip = None
            while self.prepender in hooks: hooks.remove(self.prepender)
        elif args.OnOff == "on": # Register the prepender
            self._autodip = "%%dip"
            if self.prepender not in hooks: hooks.append(self.prepender)

        print(f"Auto Execution on DDP group: {'on' if self._autodip else 'off'}", flush=True)

    def _stopdip(self, line=''):
        if self.ddp:
            Config.Verbose and print("Shutting down cluster....", flush=True, file=sys.stderr)
            self.ddp.shutdown_cluster()
            self.ddp = None

    @magic_arguments()
    @argument('-g', '--gpus', dest='gpus', type=str, nargs='+', help="list of GPU ids, or 'all' to specify all GPUs available.")
    @argument('-a', '--app', dest='appname', default=Config.DefaultApp, type=str)
    @argument('-r', '--restart', dest='restart', action="store_const", const=True, help="Restart the ipyparallel cluster.")
    @argument('-k', '--kill', dest='kill', action="store_const", const=True, help="Kill the ipyparallel cluster.")
    @argument('-i', '--info', dest='info', action="store_const", const=True, help="Show current configuration info.")
    @argument('-v', '--verbose', dest='verbose', nargs='?', type=str, const="True", choices=["True", "False"],help='print a message at each execution.')
    @line_magic
    def makedip(self, line=''):
        '''%makedip -- Setup/tear down the cluster, a DDP training group on top of it, and an app client to the DDP. '''
        if self.shell is None: raise RuntimeError("%%dip: Not in an ipython Interactive shell!")
        args = parse_argstring(self.makedip, line)

        if args.verbose: Config.Verbose = args.verbose == "True"
        if args.info: print(self.info())

        if args.kill or args.restart: self._stopdip()

        if args.kill: return

        if not self.ddp: self.init_ddp()
        self.ddp.set_verbose(Config.Verbose)

        if args.gpus:
            gpus = self.gpu_str2list(args.gpus)

            if self.ddp.ddp_group == gpus: # Group desired is already there?
                print(f"%makedip DDP group unchanged, GPUs ids: {gpus}")
                return  # same group or empty group => do nothing
            
            self.ddp.exit_group() # Exit old DDP group if exists
                
            self.ddp.new_group(gpus=gpus, appname=args.appname)
            self.shell.run_line_magic("pxconfig", "--verbose" if Config.Verbose else "--no-verbose")        

    @magic_arguments()
    @argument('-q', '--quiet', dest='quiet', action='store_true', default=False, help="Display any stdout only after task is finished, skip all the transient, real-time output.")
    @argument('-t', '--to', dest='where', nargs=None, type=str, choices=["gpu", "local", "both"], default="gpu", help="Where to run the cell.")
    @argument('-p', '--push', dest='push_vars', type=str, nargs='+', help="List of notebook variables (local namespace) to push to the DDP group, before execution of the cell.")
    @line_cell_magic
    def dip(self, line, cell=None):
        '''%%dip - Parallel execution on cluster, allows transient output be displayed.'''
        if self.shell is None: raise RuntimeError("%%dip: Not in an ipython Interactive shell!")
        assert self.ddp and self.ddp.ddp_group, "%%dip: DDP group does not exist yet.  Have you run %ddprep -g <gpu list>?"

        args = parse_argstring(self.dip, line)

        # '--push var1 var2 ...': Export variables from local namespace to DDP group.
        if args.push_vars:
            push_dict = { varname: self.shell.user_ns[f"{varname}"] for varname in args.push_vars }
            Config.Verbose and print(f"Pushing parameters to engines namespace: {args.push_vars}", flush=True)
            self.ddp.push(push_dict)

        if cell is None: return

        # '--to [gpu|local|both]': Where to run the cell
        gpus = self.ddp.ddp_group
        run_on = { 'gpu': ' ', 'both' : 'local ipython and ', 'local' : 'local ipython ONLY, SKIPPING '}
        where = f"{run_on.get(args.where,'')}" + f"DDP group (GPUs: {gpus})"
        if Config.Verbose: print(f"%%dip {line}: Running cell on {where}", flush=True)

        if (args.where == 'local') or (args.where == 'both'):
            # To run a cell inside "%%dip" locally, insert a "stop sign" in the first line,
            # it'll then bypass any %autodip cell transformer, and won't end up here again.
            self.shell.run_cell(f"{DdpMagic._no_mod}\n"+cell, silent = args.quiet)
            if args.where == 'local': return

        self.ddp.run_cell(cell, gpus=gpus, quiet=args.quiet)
        
    @line_magic
    def dipper(self, line:str=''):
        if line: self.shell.push({line.split(None,1)[0] : self.ddp})
        else: print("Usage: '%dipper <variable name>' to store the DDP object.", file=sys.stderr)

def unload_ipython_extension(ipython):
    ipython.magics_manager.registry.pop('DdpMagic')

def load_ipython_extension(ipython):
    ipython.register_magics(DdpMagic)
