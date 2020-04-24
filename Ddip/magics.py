import sys, time, os, re, ipyparallel, torch, atexit, IPython, dataclasses
from collections import OrderedDict
from typing import List
from types import SimpleNamespace
from torch.distributed import *
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic, line_cell_magic
from IPython.core.display import clear_output
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring, kwds
from ipyparallel import AsyncResult
from .ddp import Ddp

Config = SimpleNamespace(Verbose = False, DefaultApp = 'fastai_v1')

@magics_class
class DdpMagic(Magics):
    '''An iPython extension to harness the pytorch distributed data parallel (DDP)
       execution over the ipyparallel cluster.
    '''
    _no_mod = "#dip_locally"

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
        '''Prepend automatic DDP execution cell magic, unless the cell begins with '%',
        or with a known marker, to avoid infinite loop.
        See https://github.com/jdanbrown/potoo/blob/master/potoo/default_magic_magic.py'''
        if self._autodip and lines and (not lines[0].startswith('%')) and (not lines[0].startswith(DdpMagic._no_mod)):
            lines.insert(0, f"{self._autodip}\n")
        return lines

    @magic_arguments()
    @argument('-a', '--args', dest='dip_quoted_args', type=str, nargs=None, help="In '--args --in quoted_string', flags and arguments to pass to %%dip.")
    @argument('OnOff', type=str, choices=["on", "off"], nargs='?', help="Turn on auto-%%dip for the cells after this one.")
    @line_magic
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
            if args.dip_quoted_args: self._autodip += " " + args.dip_quoted_args.replace('"', '')
            if self.prepender not in hooks: hooks.append(self.prepender)

        print(f"Auto Execution on DDP group: {'on, will run cell as ' + self._autodip if self._autodip else 'Off'}", flush=True)

    @magic_arguments()
    @argument('-g', '--gpus', dest='gpus', type=str, nargs='+', help="list of GPU ids to form the DDP group. Use 'all' to specify all available GPUS.")
    @argument('-a', '--app', dest='appname', default=Config.DefaultApp, type=str)
    @argument('-r', '--restart', dest='restart', action="store_const", const=True, help="Restart the ipyparallel cluster.")
    @argument('-k', '--kill', dest='kill', action="store_const", const=True, help="Kill the ipyparallel cluster.")
    @argument('-i', '--info', dest='info', action="store_const", const=True, help="Show current configuration info.")
    @argument('-v', '--verbose', dest='verbose', nargs='?', type=str, const="True", choices=["True", "False"],help='print a message at each execution.')
    @argument('-t', '--timeout', dest='timeout', type=int, help="timeout amount to wait for engines to connect.")
    @line_magic
    def makedip(self, line=''):
        '''%makedip -- Setup/tear down the cluster, a DDP training group on top of it, and an app client to the DDP. '''
        if self.shell is None: raise RuntimeError("%%dip: Not in an ipython Interactive shell!")
        args = parse_argstring(self.makedip, line)

        if args.verbose: Config.Verbose = args.verbose == "True"
        if args.info: print(self.info())
        if args.kill or args.restart:
            Ddp.shutdown(self.ddp)
            self.ddp = None
            if args.kill: return

        if not self.ddp: self.init_ddp()
        self.ddp.set_verbose(Config.Verbose)

        if args.gpus:
            gpus = self.gpu_str2list(args.gpus)
            self.ddp.exit_group() # Exit old DDP group if exists
            if getattr(args, "timeout", None) is not None:
                ddp_kwargs = { "engine_wait" : args.timeout}
                print(f"Setting engine timeout to {args.timeout}", flush=True)
            self.ddp.new_group(gpus=gpus, appname=args.appname, **ddp_kwargs)
            self.shell.run_line_magic("pxconfig", "--verbose" if Config.Verbose else "--no-verbose")        

    @magic_arguments()
    @argument('push_vars', type=str, nargs='+', help="Push a list of variables from local ipython/notebook namespace to the DDP group processes.")
    @line_magic
    def dipush(self, line=''):
        args = parse_argstring(self.dipush, line)
        if args.push_vars:
            push_dict = { varname: self.shell.user_ns[f"{varname}"] for varname in args.push_vars }
            Config.Verbose and print(f"Pushing parameters to DDP namespace: {args.push_vars}", flush=True)
            self.ddp.push(push_dict)

    @magic_arguments()
    @argument('-r', dest='rank', type=int, default=0, help="Rank of the process to fetch the variables from.  Default to RANK 0.")
    @argument('pull_vars', type=str, nargs='+', help="Pull a list of variables from the DDP group processes to local ipython/notebook namespace.")
    @line_magic
    def dipull(self, line=''):
        args = parse_argstring(self.dipull, line)
        Config.Verbose and print(f"Pulling from DDP namespace: {args.pull_vars}", flush=True)
        remote_dict = self.ddp.pull(args.pull_vars, rank=args.rank)
        self.shell.user_ns.update(remote_dict)

    @magic_arguments()
    @argument('-q', '--quiet', dest='quiet', action='store_true', default=False, help="Display any stdout only after task is finished, skip all the transient, real-time output.")
    @argument('-S', '--see', dest='see', type=str, nargs='+', help="display outputs from process specified by a list of ranks, or 'all'. Default to 0.")
    @argument('-a', '--appcmd', dest='appcmd', type=str, help="Pass a command to the app.")
    @argument('where', nargs='?', type=str, choices=["remote", "local", "everywhere"], default="remote", help="Where to run the cell, default is remote.")
    @cell_magic
    def dip(self, line, cell):
        '''%%dip - execution in local or remote, or both namespaces.  If remote, the transient output be streamed to the notebook console.'''
        if self.shell is None: raise RuntimeError("%%dip: Not in an ipython Interactive shell!")
        if cell == "": return

        args = parse_argstring(self.dip, line)

        where = { args.where } if args.where != "everywhere" else { "local", "remote" }

        if "local" in where:
            # Insert magic string in the first line, to suppress %autodip cell mods, and prevent infitely looking back here again.
            Config.Verbose and print(f"%%dip {line}: Running cell in local namespace.", flush=True)
            self.shell.run_cell(f"{DdpMagic._no_mod}\n"+cell, silent = args.quiet)
        
        if "remote" in where:
            if self.ddp is None:
                Config.Verbose and print("%%dip: No DDP group exists to execute the cell.  Have you run %makedip ?")
                return
            gpus = self.ddp.ddp_group
            see_outputs = list(filter(lambda x: x in gpus, self.gpu_str2list(args.see))) if args.see else None
            if args.appcmd: self.ddp.app_run_cmd(args.appcmd)
            Config.Verbose and print(f"%%dip {line}: Running cell in remote DDP namespace (GPUs: {gpus}).", flush=True)
            self.ddp.run_cell(cell, gpus=gpus, quiet=args.quiet, see=see_outputs)
        
    @line_magic
    def dipper(self, line:str=''):
        if line: self.shell.push({line.split(None,1)[0] : self.ddp})
        else: print("Usage: '%dipper <variable name>' to store the DDP object.", file=sys.stderr)

def unload_ipython_extension(ipython):
    ipython.magics_manager.registry.pop('DdpMagic')

def load_ipython_extension(ipython):
    ipython.register_magics(DdpMagic)
