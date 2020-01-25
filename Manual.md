A user guide to `ddipp`

## Introduction

`ddipp` is a collection of iPython line and cell magics, only useful within an interactive iPython or Jupyter notebook session.  It uses `ipyparallel` to manage PyTorch's Distributed Data Parallel (DDP) process group as a cluster.  Each process in a DDP group is assigned to manage a particular GPU device.  In this aspect, `ddipp` is like a traffic controller, to direct the execution of cells to the DDP group when asked to, and it will route back the outputs.

`ddipp` treats `fastai` as a client application that uses the PyTorch DDP.  User can specify what application to prepare for, when creating a DDP group (using `%makedip`).  `fastai_v1` is the default application.  In this context, `ddipp` dynamically patches up `fastai` to achieve correct execution, while maintaining the interactive user experience of the typical `fastai` Jupyter notebook workflow.

## Using `ddipp`

### Installation
1. To load the extension, simply run this in iPython/Jupyter:

    `
    %load_ext ddipp
    `
### Commands/Magics in iPython/Jupyter Notebook

* Starting/Stopping/Restarting a Distributed Data Parallel (DDP) group:

    ```
    # Use all available GPUs, initialize fastai library to run in notebook, turn on verbose output

    %makedip -g all -a fastai_v1 --verbose True

    ```

* Pushing variables from the notebook "local" namespace, to the DDP group namespace:

    ```
    %dip --push var1 var2 var3..... varN
    ```

* Execute a notebook cell in parallel across the DDP processes (each has its own designated GPU):
    ```
    %%dip # Add this to the first line of a notebook cell
    <code to be run in parallel on multiple processes>
    ```

* Execute a cell in **both** the notebook and across DDP processes:
    ```
    %%dip --to both

    # Execute this cell first in local notebook, then in parallel across DDP processes
    
    from fastai.vision import *
    import os
    ```

* Turn on/off automatic parallel/DDP execution:
    ```
    %autodip on [optional flags passed to %%dip]
    ```
    ```
    < %%dip [optoinal flags] is prepended to the cells implicitly, and will be executed in the remote DDP group >

    [Subsequent cells will run in parallel, across the DDP processes]
    ```

* To halt the automatic parallel/DDP exeuction:
    ```
    %autodip off  # Subsequent cells run in local notebook namespace again.  
    ```
    or halt temporarily just for one cell when `%autodip on` is in effect:
    ```
    #dip_locally
    <Run this cell local namespace, regardless %autodip being 'on' or 'off' >
    ```


