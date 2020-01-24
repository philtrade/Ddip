# ipyparallel-torchddp


This project was inspired by 3 impactful tools by the open source community:

- The `fastai` library that comes with completely free video lessons and interactive Jupyter notebooks to speed up learning, to make programming neural network applications a breeze.

- Multiple GPUs can speed up the training of complex models, and PyTorch's Distributed Data Parallel (DDP) mode is one of the effective configuration.  To use DDP inside FastAi is quite trivial in batch script mode, **but not supported in the FastAi's interactive Jupyter notebooks**, as of late 2019.

- `ipyparallel` is a fairely mature iPython extension to manage cluster of distributed processes to handle parallel executions.

`ddipp`, an iPython extension, attempts to fill the gap, by using `ipyparallel` to facilitate aplication to use `PyTorch DDP` inside Jupyter notebooks. Specifically as a start, to support `fastai`'s course-v3 notebooks in DDP mode.


Platform tested: single host with multiple GPU, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.

To know more about iPython extensions/magics, PyTorch DDP, and ipyparallel:

- 
-
- 


## Installations

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Usage


### Load the extension:
    ```
    %load_ext ddipp
    ```
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

## Known Limitations, Issues to investigate, Features to add

* **Works in Single Host with Multiple GPU only** *#limitation #features*

    Only tested on a single host with multiple GPUs.  Both underlying components  `ipyparallel` and `PyTorch` do support distributed setup to utilize a farm/cluster of machines, each equipped with multiple GPUs.
    
    `ddipp` was designed to assist the interactive experiment in Jupyter notebook, and was developed/tested on a single host with multiple GPUs. It doesn't work in multiple hosts environment yet.  For tasks that must be deployed to a farm, they are often scripted up, and launched, harnessed in batch mode, e.g. see the  for `PyTorch` and `FastAi` DDP launch tools.

    To make `ddipp` work in distributed hosts/cluster environment would be an interesting project.  **Is it important and urgent to your work?** 

* **Process group must starts with GPU 0, and in consecutive, ascending order**  *#bug*

    In other words, on a host with GPU [0,1,2,3], these may not work properly: 
    `%makedip -g 3,2,1,0`, or `-g 1,2,3`, or `-g 0,3` etc...

* **lesson6-pets-more the cnn_learner doesn't show training speed-up with multiple GPUs** #investigate

    Most other network models do see speedups, e.g. unet_trainer, imagenet, vgg.  But this lesson notebook doesn't seem to gain at all. I/O bottleneck?  Synchronization (among GPUs) issue?
    
    * note: nvidia's `nvprof` is a powerful, low-level profiling tool to debug performance issue on multi-GPU workloads.

* **Fastai's lesson4-imdb notebook seem to train incorrectly** #investigate

    Accuracy didn't improve in the second stage of training, where a previously trained model/text embedding is loaded to train another language task.

* **Unit tests, and CI** #feature

    Although FastAi's lesson notebooks provide great integration tests, `ddipp` needs and will benefit from lightweight unit tests (especially in the collaborative open source setting.)

* **FastAi V2** #feature
    
    `ddipp` as of January 2020 works for FastAi v1's lesson notebooks, meanwhile fastai v2 is being rolled out.  If the ability to conduct multi-GPU experiments in Jupyter notebook is deemed useful, `ddipp` will need to catch up.

* **Swift for Fastai** #feature

    **Gasp!**

* **FastAi's `nbdev` Approach** #feature #investigate

    Centuries of wisdom from experimental science distilled from the many great minds, all point to the benefit of iteractive tinkering and collaborative research via documentation and tests. 
    
    In software development of this age, finally a pragmatic tool to foster one-stop development of code, documentation, and tests is emerging --- `nbdev`, the labor of love from the brains behind FastAi.

    In developing `ddipp`, the author experienced the common and often painstaking overhead of keeping the code, tests, and documentation in sync.  For future development, e.g. to add proper documentation and tests, it would be interesting to see any boost in productivity by moving to `nbdev`.





     