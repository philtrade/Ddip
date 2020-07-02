
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


