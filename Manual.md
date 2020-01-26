# User Guide to Double Dip `ddipp`

## Introduction

`ddipp` is a collection of line and cell magics for iPython and Jupyter notebook.  It uses `ipyparallel` to manage PyTorch's Distributed Data Parallel (DDP) process group as a cluster.  In this aspect, `ddipp` is like a traffic controller, it directs the execution of cells between the local process (the interactive notebook), and the remote DDP process group, and streams/display the outputs.

`ddipp` treats `fastai` as a client application that uses the PyTorch DDP (currently  `fastai_v1` is the default application).  `ddipp` is designed to minimize changes to  existing `fastai` Jupyter notebooks, and without requiring any change to the fastai codebase itself.


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
* Do some data loading, massage, exploration, augmentation etc..

* Ready to create training data that need to be on accessible to each DDP process, let us push variables from the notebook "local" namespace, to the DDP group namespace:

    ```
    %dipush var1 var2 var3..... varN
    ```

* Execute a notebook cell in parallel across the DDP processes (each has its own designated GPU):
    ```
    %%dip # Add this to the first line of a notebook cell
    <code to be run in parallel on multiple processes>
    ```

* Execute a cell in **both** the notebook and across DDP processes:
    ```
    %%dip everywhere

    # Execute this cell first in local notebook, then in parallel across DDP processes
    
    from fastai.vision import *
    import os
    ```

* Turn on/off automatic parallel/DDP execution:
    ```
    %autodip {on, off} [optional flags passed to %%dip]
    ```
    ```
    [If on, subsequent cells will run in parallel among the DDP processes]
    [If off, subsequent cells will run in the local notebook namespace]

    ```

* To run one cell locally, regardless %autodip is on or off, insert any of below as the first line of the cell:
    ```
    #dip_locally
    ```
    ```
    %%dip locally [optional flags..]
    ```


