


## Draft on Mpify ReadMe.md

### Overview 

**`mpify`** is an simple API to run function (the "target function") on a group of *ranked* processes, and is designed work in Jupyter/IPython.
 
###  Main Features
  * Functions and objects defined in the same notebook can ride along via the target function's input parameter list --- *a feature not available from the current Python `multiprocessing` or `torch.multiprocessing`*.
  * A helper `mpify.import_star()` to handle `from X import *` within a function --- *a usage banned by Python*.
  * In-n-out execution model: *spin-up -> execute -> spin-down & terminate* within a single call of `ranch()`.  The Jupyter session is the *parent process*, it spins up and down children processes.
  * Process rank [`0..N-1`] and the group size `N` are stored in `os.environ`.  **The parent Jupyter process can participate, and it does by default as the rank-`0` process**, and it will receive the return value of target function run as rank-`0`.
  * User can write custom context manager to wrap around the call to the target function in the spawned processes.  As an example, `mpify` provides `TorchDDPCtx` to run the target in PyTorch's distributed data parallel mode.

#### Example: Adapting [the `fastai v2`notebook on training `imagenette`](https://github.com/fastai/course-v4/blob/master/nbs/07_sizing_and_tta.ipynb) to run on multiple GPUs within the interactive session.  From:

<img src="/images/imagenette_07_orig.png" height="270">

To this:

<img src="/images/imagenette_07_mpified.png" height="320">

### Usage 

To adapt a function `myfunc()` to be both single process/multiprocess friendly within the notebook, make two changes, so that all the resources `myfunc()` needs are accessible when it's called.
  
Say the notebook has defined function `foo()` or `objectA`, and `myfunc()` uses them:
  
  1. First, pass `foo` and `objectA` as parameters of `myfunc()`:
  
  ```python
    def myfunc(..., foo, objectA, ...):
    
        < use foo() and objectA as before >
  ```
  2. Second, import modules that `myfunc()` needs at the top, because `myfunc()` will be run on a clean, fresh Python interpreter process.  To handle `from X import *`, use `mpify.import_star(['X'])`:
  
  E.g.: Originally the notebook may have the following:
  ```python
    # At the notebook beginning:
    from utils import *                # starred import
    import numpy as np
    import torch
    from fastai2.distributed import *  # more starred import
    
    # some cells later
    def foo():
       ...
       
    # and later
    objectA = 100
    
    def myfunc(arg1): #
        x = np.array([objectA])
        foo(x)
        ...
  ```
    
  To adapt it, one can write:
  
  ```python
    def myfunc(arg1, ..., foo, objectA, ...):
        import mpify
        mpify.import_star(['utils', 'fastai2.distributed']) # Helper to handle "from X import *"
        import numpy as np                       
        import torch                         # Other imports earlier in notebook are copied here
        
        x = np.array([objectA])
        foo(x)
        ...
        if os.environ.get('RANK', '0') == '0': return f"Rank-0 process returning"
  ```

  Then launch it away to 5 ranked processes using `mpify.ranch()`
  ```python
    import mpify
    r = mpify.ranch(5, myfunc, arg1, foo, objectA)
  ```

### A few technicalities when using `mpify`:
#### At the functions, objects, namespace level:

- Thanks to the excellent [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), functions and lambdas defined locally **within** the Jupyter notebook can be passed to spawned children processes. `multiprocess` is an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`.

#### At the process level:

- In each process, generic distributed attributes of *local rank*, *global rank*, *world size* are available in the `os.environ[LOCAL_RANK, RANK, and WORLD_SIZE]` respectively as strings (not integers).  They follow the PyTorch definition in [distributed data parallel convention](https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940).

- The foreground Jupyter session participates as rank-0 process by default, but it's not mandatory. If it does, user can use widgets such as `tqdm` and `fastprogress` to visualize the function progress.  It will also receive the whatever target function returns.

- `return`s from the spawned processes are **discarded**.  Gathering results using `multiprocess.Queue` or `shared_memory` or any fancy shmmmancy IPC isn't supported in `mpify` yet, but would be a good exercise for the existing context manager described below.


#### Resources mangement using custom context manager `ranch(... ctx=Blah ...)`:

- `mpify.ranch(ctx=YourCtxMgr)` lets user wrap around the function execution:

  spawn -> [`ctx.__enter__()`]-> run function -> [`ctx.__exit__()`] -> terminate.

- Use it to set up execution environment, manage resources, open/close database or network connection, or even plug the target function into a pipeline of functions.  Do fun and boring things.

`mpify` provides `TorchDDPCtx` to setup/tear-down of PyTorch's distributed data parallel: 

- `TorchDDPCtx` does the "bold stuff":
  
  spawn -> [**set target GPU (initialize GPU context if not already) and DDP**]-> run function -> [**cleanup DDP**] -> terminate.

- Either pass a custom `TorchDDPCtx` object to `mpify.ranch()`, or use the convenience routine `mpify.in_torchddp()`.  The two are equivalent below:

```python
    # lazy, just use the default TorchDDPCtx() setting
    result = in_torchddp(world_size, create_and_train_model, *args, kwargs*)`

    # or more explicit, can be your own context manager:

    result = ranch(world_size, create_and_train_model, *args, ctx=TorchDDPCtx(), kwargs*)
 ```

More notebook examples may come along in the future.


References:

**`mpify`** was conceived to overcome the many quirks when  {Python multiprocess, Jupyter, multiple CUDA GPUs, Windows or Unix} meet, without mucking around with complicated cluster setup and . <include link to blog when available>

* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

