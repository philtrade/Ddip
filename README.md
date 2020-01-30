# Interactive PyTorch DDP Training in FastAI Jupyter Notebooks

`Ddip` ("Dee dip") --- Distributed Data "interactive" Parallel is a little iPython extension of line and cell magics to bring together `fastai` lesson notebooks [[1]](#course_v3) and PyTorch's Distributed Data Parallel [[2]](#pytorchddp).  It uses `ipyparallel` [[3]](#ipp) to manage the DDP process group. 

Platform tested: single host with multiple Nvidia CUDA GPUs, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.

## Features:

`Ddip` was designed to make experiments with multiple GPU/Distributed Data Parallel training a little bit easier in Fastai notebooks. Towards that end:

1. Easy switching of code execution between PyTorch's multiprocess DDP group, and that in the local notebook namespace.
2. Automatic garbage collection and freeing GPU cache memory after parallel execution of a cell, thus reducing the occurence of GPU out of memory error.
3. Minimize code changes to Fastai `course v3` lesson notebooks to run in DDP -- usually 3-5 lines of iPython `%,%%` magics will do -- while keeping the look-and-feel as identical as possible.  E.g. real time progress bar, `Learner.lr_find()` works like the same as in non-distributed mode.
4. Extensible to support future versions of fastai library.  Porting to `fastai v2` is a high priority to-do.


## Installation:

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Overview and Examples:
### Quick Overview of `Ddip` line and cell magics
* `%load_ext Ddip`,  to load the extension
* `%makedip` to start/stop/restart a Distributed Data Parallel process group.  
    E.g. to initialize `fastai` to use DDP in the notebook: `%makedip -g all -a fastai_v1`
* `%%dip {remote | local | everywhere}` , to  execute a cell in DDP, local notebook, or both namespaces.
* `%autodip {on | off}`, to automatically execute subsequent cells in the DDP group, without requiring `%%dip` every time.
* `%dipush bs src func1 .....`, to push `bs`, `src`, and `func1` from notebook to the DDP processes
* `%dipll foo bar`, to pull objects named `foo` and `bar` from rank-0 DDP process into the local notebook

### Examples:
* [The `fastai` lesson3-camvid notebook using `Ddip` to train in DDP](notebooks/Ddip_usage_fastai.ipynb),
* [More notebooks](notebooks/)
## [Known Limitations, Issues, Bugs and Features to Add](Issues.md)

## References:

1. <a name="course_v3"></a> [FastAI Course v3](https://course.fast.ai/)

2. <a name="pytorchddp"></a>On Distributed training:
* [Tutorial from PyTorch on Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), 
* [Launching fastai to use DDP](https://docs.fast.ai/distributed.html), *FastAI*
* Further readings: [PyTorch Lightning -- Tips for faster training ](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565), [On the performance of different training parallelism](http://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)


3. <a name="ipp"></a>On `ipyparallel`, a few resources and inspirations to `Ddip`:
* [The official documentation](https://ipyparallel.readthedocs.io/en/latest/intro.html),
* [An intro to ipyparallel](http://activisiongamescience.github.io/2016/04/19/IPython-Parallel-Introduction/), *Activevision Game Science*
* [Using ipyparallel](http://people.duke.edu/~ccc14/sta-663-2016/19C_IPyParallel.html), *Duke University, "Computational Statistics in Python"*
* [Interactive Distributed Deep Learning with Jupyter Notebooks](https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post206s2-file3.pdf), *Lawrence Berkeley National Laboratory/Cray Inc.*



