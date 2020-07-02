


## Draft on Mpify ReadMe.md

### Overview 

**Mpify** is an simple API to run function (the "target function") on a group of *ranked* processes, and is designed work in Jupyter/IPython.  User can pass functions and objects defined in the notebook to the target function, and the Jupyter process can participate in the group as the rank-0 process.

User can customize/enrich its default simplistic behavior with custom context manager, e.g. to manage resources, distribute/collect results in rank-0 process, etc.

E.g. Adapting [`fastai2` imagenette training notebook](), the original code for training on a single single process:

![Original 07_sizing_and_tta.ipynb](/images/imagenette_07_orig.png | height=200)

vs and the adapted with `mpify` to run on 3 processes:

![Adapted 07_sizing_and_tta.ipynb](/images/imagenette_07_mpified.png | height=200)

###  Main features
  * Functions and objects defined in the same notebook can ride along via the target function's input parameter list --- *a feature not available from the current Python `multiprocessing` or `torch.multiprocessing`*.
  * A helper `Mpify.import_star()` to handle `from X import *` within a function --- *a usage banned by Python*.
  * In-n-out execution model: *spin-up -> execute -> spin-down & terminate* within a single call of `ranch()`.  The Jupyter session is the *parent process*, it spins up and down children processes.
  * Process rank [`0..N-1`] and the group size `N` are stored in `os.environ`.  **The parent Jupyter process can participate, and it does by default as the rank-`0` process**, and it will receive the return value of target function run as rank-`0`.

  * User can provide a context manager to wrap around the target function execution: to manage resources and setup/teardown execution environment etc.. `Mpify` provides `TorchDDPCtx` to illustrate the setup/tear-down of PyTorch's distributed data parallel training..

**Mpify** was conceived to overcome the many quirks of *getting Python multiprocess x Jupyter x multiple CUDA GPUs on {Windows or Unix}* to cooperate. <include link to blog when available>


### Usage 

  * Say the notebook has defined function `foo()` or `objectA`, and `dist_func()` needs them, make sure the two appear as:
    ```python
    def dist_func(... foo, objectA...):
        < function body >
    ```
  * Let's not forget imported modules that `dist_func()` needs. 
  
    Import them inside `dist_func()` at the top. Use `Mpify.import_star(['X'])` to handle `from X import *`.
  
    E.g.: the notebook has these  <use screenshot>
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

`Mpify` provides `TorchDDPCtx` to illustrate the setup/tear-down of PyTorch's distributed data parallel: 

- `TorchDDPCtx` does the "bold stuff":
  
  spawn -> [**set target GPU (initialize GPU context if not already) and DDP**]-> run function -> [**cleanup DDP**] -> terminate.

- Either pass an `TorchDDPCtx` object to `ranch()`, or use the convenience routine `Mpify.torchddp_launch()`.  The two are equivalent below:

  * ```python
    # lazy, just use the default TorchDDPCtx() setting
    result = torchddp_launch(world_size, create_and_train_model, *args, kwargs*)`

    # or more explicit, can be your own context manager:

    result = ranch(world_size, create_and_train_model, *args, ctx=TorchDDPCtx(), kwargs*)

  - Loading/saving model states in DDP need to follow a couple guidelines [add link] to avoid file corruption/race conditions.  In the examples, fortunately `fastai` already takes care of that.


References:
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3

