import sys, time, os, re, ipyparallel, torch, atexit, IPython, dataclasses
from collections import OrderedDict
from typing import List
from torch.distributed import *
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython.core.display import clear_output
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from ipyparallel import AsyncResult
from .torchDDP import Ddp

this = sys.modules[__name__]
this.defaults = { 'appname' : 'fastai-v1' }

Debug = True
Verbose = True

def _debug(*args, **kwargs):
    if Debug: print(*args, file=sys.stderr, **kwargs)

@magics_class
class IppDdp(Magics):
    '''IppDdp is an ipython extension of line and cell magics to harness the pytorch
    distributed data parallel (DDP) execution over the ipyparallel cluster.
    '''
    class StreamPrinter():
        ''' Each invocation prints from where it left off till the end of the current output '''
        def __init__(self, streams, *args, **kwargs):
            self._counts = [0] * len(streams)
        def __call__(self, streams, *args, **kwargs):
            for i, st in enumerate(streams):
                if (len(st) > self._counts[i]):
                    print(st[self._counts[i]:], end='', flush=True)
                    self._counts[i] = len(st)

    _default_streaming_pause = 0.1

    def __init__(self, shell:IPython.InteractiveShell, **kwargs):
        super(IppDdp, self).__init__(shell=shell) # will setup self.shell
        self._streaming_pause = IppDdp._default_streaming_pause
        self.init_ddp(**kwargs)

    def init_ddp(self, **kwargs):
        '''Initialize a new DDP group, reset states of ipython magic, and associated applicaiton to None.'''
        self._autoddp = None # Flag to control if parallel execution is by default ON or OFF
        self.ddp = Ddp(**kwargs) # Controller for DDP, and the underlying ipyparallel cluster

    def __del__(self):
        if self.ddp: self.ddp.shutdown_cluster()

    def prepender(self, lines:List[str]):
        ''' https://github.com/jdanbrown/potoo/blob/master/potoo/default_magic_magic.py'''
        if self._autoddp and lines and (not lines[0].startswith('%')):
            lines.insert(0, f'%%{self._autoddp}\n')
        return lines
        
    @line_magic
    def autoddp(self, line:str):
        '''Prepend %%ddpx to subsequent cells so that they will run on the distributed data parallel cluster.'''
        '''Todo: free memory after each run'''
        if line:
            hooks = self.shell.input_transformers_cleanup
            if line.split(None,1)[0] == "off": # Unregister the prepender
                self._autoddp = None
                while self.prepender in hooks: hooks.remove(self.prepender)
            else: # Register the prepender, 
                self._autoddp = line
                if self.prepender not in hooks: hooks.append(self.prepender)
        print(f"Auto parallel execution: {self._autoddp if self._autoddp else 'Off'}")
        return self._autoddp

    @line_magic
    def ddpstop(self, line=''):
        if self.ddp:
            Verbose and print("%ddpstop shutting down cluster....", flush=True, file=sys.stderr)
            self.ddp.shutdown_cluster()
            self.ddp = None

    @magic_arguments()
    @argument('-g', '--gpus', dest='gpus', type=str, nargs='+', help="comma or space seperated list of GPU ids, or 'all' to specify all GPUs available.")
    @argument('-a', '--app', dest='appname', default=this.defaults['appname'], type=str)
    @argument('-r', '--restart', dest='restart', action="store_const", const=True, help="Restart all engine processes.")
    @argument('-d', '--debug', nargs='?', type=lambda x: (str(x).lower() == 'true'))
    @argument('-v', '--verbose', nargs='?', type=lambda x: (str(x).lower() == 'true'))
    @line_magic
    def ddprep(self, line=''):
        '''%ddprep -- line magic to setup/tear down the cluster as a DDP training group, app-specific handling of object'''
        '''Todo: add -i for info on current config'''
        if self.shell is None: raise RuntimeError("%%ddpx: Not in an ipython Interactive shell!")
        args = parse_argstring(self.ddprep, line)

        global Debug # Monkey hack until  classes have their own module namespaces and own debug/verbose flags
        Debug = args.debug
        global Verbose
        Verbose = args.verbose

        if args.restart: self.ddpstop()
        if not self.ddp: self.init_ddp()

        if args.gpus:
            if 'all' in args.gpus: gpus = list(range(torch.cuda.device_count()))
            else: gpus = list(OrderedDict.fromkeys([ int(g) for g in re.split(r',\s*|\s+', ','.join(args.gpus))]))

            if self.ddp.ddp_group == gpus: # Group desired is already there?
                print(f"%ddprep DDP group unchanged, GPUs ids: {gpus}")
                return  # same group or empty group => do nothing
            
            self.ddp.exit_group() # Exit old DDP group if exists
                
            self.ddp.new_group(gpus=gpus, appname=args.appname)
 
    @magic_arguments()
    @argument('--quiet', dest='quiet', action='store_true', help="Display any stdout only after task is finished, skip all the transient, real-time output.")
    @cell_magic
    def ddpx(self, line, cell): # CAN WATCH BE CONTEXT SENSITIVE?
        '''%%ddpx - Parallel execution on cluster, allows transient output be displayed'''
        '''Todo: add --free to free cuda cache and calls gc.collect() after the cell is run'''
        if self.shell is None: raise RuntimeError("%%ddpx: Not in an ipython Interactive shell!")
        assert self.ddp and self.ddp.ddp_group, "%%ddpx: DDP group does not exist yet.  Have you run %ddprep?"
        Verbose and print(f"Invoking cell magic: %%ddpx {line}", file=sys.stderr, flush=True)

        args = parse_argstring(self.ddpx, line)
        px_args=f"--noblock --targets {self.ddp.gpus_str()}"
        try:
            ar = self.shell.run_cell_magic("px", px_args, cell) # use parallel_execute?
            watcher = IppDdp.StreamPrinter(ar.stdout) if (not args.quiet) else None

            if watcher:
                while not ar.ready(): # Simulate wait on blocking execution
                    watcher(ar.stdout)
                    time.sleep(self._streaming_pause)
                watcher(ar.stdout)
                clear_output()

            r = ar.get() # Blocks till completion
        except KeyboardInterrupt:
            Verbose and print(f"Caugth interrupt, sending SIGINT to engines....", file=sys.stderr, flush=True)
            self.ddp.cluster.interrupt_engines(self.ddp.ddp_group)

        ar.display_outputs()
        return r

    @line_magic
    def ddpobj(self, line:str=''):
        if line: self.shell.push({line.split(None,1)[0] : self.ddp})
        else: print("%ddpobj requires one argument: a variable name to store the DDP object.", file=sys.stderr)

def unload_ipython_extension(ipython):
    ipython.magics_manager.registry.pop('IppDdp')

def load_ipython_extension(ipython):
    ipython.register_magics(IppDdp)
