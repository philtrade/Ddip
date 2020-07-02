


## Draft on Mpify ReadMe.md

### Overview 

**`mpify`** is an simple API to run function (the "target function") on a group of *ranked* processes, and is designed work in Jupyter/IPython.  User can pass functions and objects defined in the notebook to the target function, and the Jupyter process can participate in the group as the rank-0 process.

User can customize/enrich its default simplistic behavior with custom context manager, e.g. to manage resources, distribute/collect results in rank-0 process, etc.

E.g. Adapting [the `fastai v2`notebook on training `imagenette`](https://github.com/fastai/course-v4/blob/master/nbs/07_sizing_and_tta.ipynb) to run on multiple GPUs within the interactive session.  From:

<img src="/images/imagenette_07_orig.png" height="450">

To this:

<img src="/images/imagenette_07_mpified.png" height="400">

###  Main features
  * Functions and objects defined in the same notebook can ride along via the target function's input parameter list --- *a feature not available from the current Python `multiprocessing` or `torch.multiprocessing`*.
  * A helper `mpify.import_star()` to handle `from X import *` within a function --- *a usage banned by Python*.
  * In-n-out execution model: *spin-up -> execute -> spin-down & terminate* within a single call of `ranch()`.  The Jupyter session is the *parent process*, it spins up and down children processes.
  * Process rank [`0..N-1`] and the group size `N` are stored in `os.environ`.  **The parent Jupyter process can participate, and it does by default as the rank-`0` process**, and it will receive the return value of target function run as rank-`0`.

  * User can provide a context manager to wrap around the target function execution: to manage resources and setup/teardown execution environment etc.. `mpify` provides `TorchDDPCtx` to illustrate the setup/tear-down of PyTorch's distributed data parallel training..

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available>


### Usage 

  * Say the notebook has defined function `foo()` or `objectA`, and `myfunc()` needs them.  To adapt `myfunc()` to run on multiple processes within the notebook, make two changes, so that all the resources `myfunc()` needs are accessible when it's called:
  
  1. First, pass `foo` and `objectA` as parameters of `myfunc()`:
  
  ```python
    def myfunc(..., foo, objectA, ...):
    
        < use foo() and objectA as before >
  ```
  2. Second, import modules that `myfunc()` needs at the top, because `myfunc()` will be run on a clean, fresh Python interpreter process.  To handle `from X import *`, use `mpify.import_star(['X'])`:
  
  E.g.: Originally the notebook may have the following:
  ```
    # At the notebook beginning:
    from utils import *
    from fancy import *
    import numpy as np
    import torch
    from torch.distributed import *
    
    # some cells later
    def foo():
       ...
       
    # and later
    objectA = 100
    
    def func_happy_as_single_process(arg1): #
        x = np.array([objectA])
        foo(x)
        ...
  ```
    
  To adapt `func_happy_as_single_process` to be multiprocess-friendly, one can write:
  
  ```python
    def new_func(arg1, foo, objectA **kwargs):
        from Mpify import import_star      # Helper to handle "from X import *" syntax
        import_star(['utils', 'fancy'])
        import numpy as np                 # Imports earlier in notebook are copied here.
        import torch
        import_star(['torch.distributed'])
        
        x = np.array([objectA])
        foo(x)
        ...
        if os.environ.get('RANK', '0') == '0': return f"Rank-0 process returning"
  ```

  3. Launch it to 5 ranked processes:
  ```python
    import mpify
    r = mypify.ranch(5, new_func, arg1, foo, objectA)
  ```

### A few technicalities when using `mpify`:
#### At the functions, objects, namespace level:
- Functions and lambdas defined locally **within** the Jupyter notebook can be passed to spawned children processes, thanks to the excellent [`multiprocess` libray](https://github.com/uqfoundation/multiprocess), an actively supported alternative to Python's default `multiprocessing` and `torch.multiprocessing`.  Simply list them as target function parameters.

#### At the process level:

- In each process, generic distributed attributes such as *local rank*, *global rank*, *world size* will be passed to `os.environ` dictionary as `LOCAL_RANK, RANK, and WORLD_SIZE` (note: values are strings, not integers in `os.environ`).  Their definitions follow the PyTorch's [distributed data parallel convention](https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940).

- The interactive Jupyter session itself participates as rank-0 process by default.  Because it runs in the foreground, user can use widgets such as `tqdm` and `fastprogress` to visualize the function progress.

- Upon completion, the Jupyter process can receive whatever the function returns from this rank's execution, but `return`s from the spawned processes are **discarded**.  Gathering results using `multiprocess.Queue` or `shared_memory` isn't supported in `mpify` yet, but would be a good exercise for the existing context manager mechanism described below.


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
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

