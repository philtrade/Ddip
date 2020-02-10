# Interactive PyTorch DDP Training in FastAI Jupyter Notebooks

`Ddip` ("Dee dip") --- Distributed Data "interactive" Parallel is a little iPython extension of line and cell magics to bring together `fastai` lesson notebooks [[1]](#course_v3) and PyTorch's Distributed Data Parallel [[2]](#pytorchddp).  It uses `ipyparallel` [[3]](#ipp) to manage the DDP process group. 

Platform tested: single host with multiple Nvidia CUDA GPUs, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.

## Features:

    "Distributed training doesn’t work in a notebook..."
>>>>-- *FastAI's tutorial on [How to launch a distributed training](https://docs.fast.ai/distributed.html)*

`Ddip` was conceived to address the above, with the following features:

1. Switch execution easily between PyTorch's multiprocess DDP group and local notebook namespace.

2. Takes 3 - 5 lines of iPython magics to port a Fastai `course v3` notebook to train in DDP.

3. Reduce chance of GPU out of memory error by automatically emptying GPU cache memory after executing a cell in the GPU proc.

4. Extensible architecture.  Future support for `fastai v2` could be implemented as a loadable module, like that for `fastai v1`.

Summary of [*speedup observed in FastAI notebooks when trained with 3 GPUs*](docs/speedups_dl1.md).

## Installation:

Current version: 0.1.0

`pip install git+https://github.com/philtrade/Ddip.git@v0.1.0#egg=Ddip`

## Overview:
### `Ddip` provides these iPython line and cell magics:
* `%load_ext Ddip`,  to load the extension.
* `%makedip ...` to start/stop/restart a DDP group, and initialize a module such as `fastai_v1`.  
* `%%dip ...` , to execute a cell in the DDP group, or local notebook, or both.
* `%autodip ...`, to execute subsequent cells in the DDP group, without requiring `%%dip` every time.
* `%dipush`, and `%dipull`, to pass things between the notebook and the DDP namespaces.


## How to run DDP with in FastAI notebooks with `Ddip`:
* [Distributed Training in `fastai` Notebook using `Ddip` - a tutorial](notebooks/Ddip_usage_fastai.ipynb)
* Example notebooks of `Ddip` iPython magics:
    - [`%makedip`](notebooks/usage_%25makedip.ipynb)
    - [`%%dip` `%autodip`](notebooks/usage_%25%25dip_%25autodip.ipynb)
    - [`%dipush` `%dipull`](notebooks/usage_%25dipush_%25dipull.ipynb).
* [More Notebooks](notebooks/)

## [Known Issues and Room for Improvements](Issues.md)

## References:

1. <a name="course_v3"></a> [FastAI Course v3](https://course.fast.ai/)

2. <a name="pytorchddp"></a>On Distributed Training:
* [Tutorial from PyTorch on Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [Launching fastai to use DDP](https://docs.fast.ai/distributed.html), *FastAI*
* Further readings: [PyTorch Lightning -- Tips for faster training ](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565), [On the performance of different training parallelism](http://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)


3. <a name="ipp"></a>On `ipyparallel`:
* [The Official ipyparallel Documentation](https://ipyparallel.readthedocs.io/en/latest/intro.html)
* [An Intro to ipyparallel](http://activisiongamescience.github.io/2016/04/19/IPython-Parallel-Introduction/), *Activevision Game Science*
* [Using ipyparallel](http://people.duke.edu/~ccc14/sta-663-2016/19C_IPyParallel.html), *Duke University, "Computational Statistics in Python"*
* [Interactive Distributed Deep Learning with Jupyter Notebooks](https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post206s2-file3.pdf), *Lawrence Berkeley National Laboratory/Cray Inc.*



