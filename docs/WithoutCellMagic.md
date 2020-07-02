


## Draft on Mpify ReadMe.md

### Overview 

**Mpify** is an simple API to run function (the "target function") on a group of distributed, *ranked* processes in parallel.  In particular, it works inside Jupyter notebook, interleaving with the usual interactive tasks, without mucking around complicated cluster/process pool configuration and state maintenance, and thrashing between the Jupyter session and terminal shell keystrokes.

Adapt an existing or write a new function to include non-local things it needs in the parameter list, and import modules at the function beginning, then launch it with `Mpify.ranch()`:

<notebook snapshot 1>

    ```python
    import Mpify

    r = Mpify.ranch(N, dist_func, x, y, helperA, objB, name="mpified!")
    ```

###  Main features
  * Functions and objects defined in the same notebook can ride along via the target function's input parameter list --- *a feature not available from the current Python `multiprocessing` or `torch.multiprocessing`*.
  * A helper `Mpify.import_star()` to handle `from X import *` within a function --- *a usage banned by Python*.
  * In-n-out execution model: *spin-up -> execute -> spin-down & terminate* within a single call of `ranch()`.  The Jupyter session is the *parent process*, it spins up and down children processes.
  * Each process has its unique rank [`0..N-1`] and the group size `N` stored in `os.environ`.  **The parent Jupyter process can participate, and it does by default as rank-`0`**, and will receive whatever its target function `return` when run as rank-`0`.

  * User can provide a context manager to claim/clean-up resources before/after the target function is run.

Illustrate usage with screen shot.

Example use case: adapt an existing `fastai2` notebook to train on multiple GPU on-demand:
< snapshot 2>

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available>


### Usage 

  * if the notebook has defined function `foo()` or `objectA`, and `dist_func()` needs them, make sure the two appear as:
    ```python
    def dist_func(... foo, objectA...):
        < function body >
    ```
  * if the notebook has already imported a bunch of modules that `dist_func()` needs, import them inside `dist_func()` as well.  Use `Mpify.import_star(['mod_A'...])` to handle `from mod_A import *`.
  
    E.g.: the notebook has imported these in earlier cells  <use screenshot>
    > ```
    > from utils import *
    > from fancy import *
    > import numpy as np
    > import torch
    > from torch.distributed import *
    > 
    > <some cells later>
    >
    > def func_happy_as_single_process(arg1): #
    >   x = np.array(....)
    > ```
    To run `func_happy_as_single_process` on 5 multiprocess, one can write a wrapper:
  
    ```python
    def dist_func(single_fn, *args, **kwargs):
        from Mpify import import_star
        import_star(['utils', 'fancy'])
        import numpy as np
        import torch
        import_star(['torch.distributed'])

        if os.environ.get('RANK', '0') == '0': # os.environ are all strings not integers!
            print(f"Rank-0 process here")
        return single_fn(*args, **kwargs)
    ```

    and launch the 5 ranked processes away:
    ```python
    r = Mypify.ranch(5, dist_func, func_happy_as_single_process, arg1)
    ```

#### At the functions, objects, namespace level, using `Mpify`:
- Functions and lambdas defined locally **within** the Jupyter notebook can be easily passed to the spawned children processes.  Thanks to the excellent `multiprocess` module [ add link ], an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`.  Just pass them explicitly to the target function as input parameters.

#### At the process level:

- In each process, generic distributed attributes such as *local rank*, *global rank*, *world size* will be passed to `os.environ` dictionary as `LOCAL_RANK, RANK, and WORLD_SIZE` (note: values are strings, not integers in `os.environ`).  The definition of these follow the PyTorch's distributed data parallel convention <add link>.

- The interactive Jupyter session itself participate as rank-0 by default.  Because it runs in the foreground, user can use widgets such as `tqdm` and `fastprogress` to visualize the function progress.

- Upon completion, the Jupyter process can receive whatever the function returns from this rank's execution, but `return`s from the spawned processes are **discarded**.  Use `multiprocess.Queue` and `shared_memory`, if fancy-shhhmancy IPC is needed.

#### Resources mangement using custom context manager `ranch(... ctx=Blah ...)`:

- `Mpify.ranch(ctx=YourCtxMgr)` lets user wrap around the function execution:

  spawn -> [`ctx.__enter__()`]-> run function -> [`ctx.__exit__()`] -> terminate.

- Use it to set up execution environment, manage imports, open/close database or network connection, or even plug the target function into a pipeline of functions.

`Mpify` provides `TorchDDPCtx` to illustrate the setup/tear-down of PyTorch's distributed data parallel process group.  Such a long winding name, I know. Let's call it DDPPG. 

- `TorchDDPCtx` does the "bold stuff":
  
  spawn -> [**set target GPU (initialize GPU context if not already) and DDP**]-> run function -> [**cleanup DDP**] -> terminate.

- Either pass an `TorchDDPCtx` object to `ranch()`, or use the convenience routine `Mpify.torchddp_launch()`.  The two are equivalent below:

  * ```python
    # lazy, just use the default TorchDDPCtx() setting
    result = torchddp_launch(world_size, create_and_train_model, *args, kwargs*)`

    # or more explicit, can be your own context manager:

    result = ranch(world_size, create_and_train_model, *args, ctx=TorchDDPCtx(), kwargs*)

  - Loading/saving model states in DDP need to follow a couple guidelines [add link] to avoid file corruption/race conditions.  In the examples, fortunately `fastai` already takes care of that.


----



### Drafts on blog: "On Playing Nice Together: Quirks in Python Multiprocessing with Multiple GPUs, in Jupyter Notebooks"

To harness multiprocess in Python has its share of complexity, in particular, resources management across process boundaries. The matter is further complicated by Jupyter's interactive model (namespace), and the unique characteristics of new breed of important resources such as GPU drivers.

*Python namespace inheritence, existing opened resources, background pickling of Python objects, OS-dependency*, to name a few, lead to somewhat complicated choice, and subsequent usage of `spawn` vs `fork` starting method, provided by the `multiprocessing` and `torch.multiprocessing` modules:

- `fork` inherits existing parent process states when child process starts.  All libraries imported, functions created are accessible automagically, nice for Jupyter/IPython, right?  Not so fast:
  * Windows OS doesn't support fork, ooops
  
  * CUDA GPU driver, if already initialized in the parent process, cannot be shared (it will crash) in children processes, but `fork()` does exactly that -- share it with child process.  In a Jupyter notebook session, if one loads up a neural network model to a GPU to do one experiment, then decides to do multiprocess, multi-GPU distributed training, `fork()` method will crash.
  
  * How about undo the CUDA initialization before `fork`?  Killing the context, while programmatically possible (e.g. in `PyCUDA` and `Numba` via resetting device or context), it pulls the rug from underneath, and throw off ALL existing GPU tensors and data structure in the Python layer -- they are orphaned, dangling pointers, and would crash the application when accessed.  It is saying "don't do it unless you are ready to accept the inconvenient consequence."   It is perhaps the reason that the PyTorch team refuses to implement an API to reset the device/release the context.

  That leads to the other choice, the `spawn` starting method.

- `spawn` starts a clean slate in child process.
  * Child process can safely initialize new CUDA context (which takes up 600MB GPU memory in PyTorch's case, by the way).

  * but the `spawn` doesn't work well in interactive Jupyter:
    - It cannot access any locally defined python function or `lambda` defined in the parent Jupyter session.  Basically it forces users to name the lamdas and save the newly defined functions to a module file on disk, and child process must import `myfuncs` in order to use the functions.

  * because `spawn` starts a clean Python interpreter, all libraries that were needed and loaded in the Jupyter parent process, must be re-imported in the child process.  However:
    - the semantic "from somemodule import *" in a Python function is syntantically disallowed.  But processed spawned from `multiprocessing.Process()` or `multiprocessing.Pool()` runs a Python target function only.
    - it means application writers have to organize any imports/locally defined functions into a separate module file `X`, write a function `foo()` that `import X`, and spawn process with `Process(target=foo,....`
    - All these extra bookkeeping/administrative are counter-productive overheads/distractions to users engaged in the Jupyter sessions.



References:
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

