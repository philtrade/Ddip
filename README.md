# Interactive PyTorch DDP Training in FastAI Jupyter Notebooks

`Ddip` ("Dee dip") --- Distributed Data "interactive" Parallel is a little iPython extension of line and cell magics to bring together `fastai` lesson notebooks [[1]](#course_v3) and PyTorch's Distributed Data Parallel [[2]](#pytorchddp).  It uses `ipyparallel` [[3]](#ipp) to manage the DDP process group. 

Platform tested: single host with multiple Nvidia CUDA GPUs, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.

## Features:

`Ddip` was conceived to address an unfilled gap mentioned in FastAI's [How to launch a distributed training](https://docs.fast.ai/distributed.html), that:
" *Distributed training doesn’t work in a notebook ...*"


`Ddip` tries to make notebook experiments in FastAI to take advantage of multi-GPU/DDP a little easier:

1. Switch execution between PyTorch's multiprocess DDP group and local notebook namespace with ease.

2. Automatic `gc.collect()` and `torch.cuda.empty_cache()` after parallel execution to avoid OOM.

3. Usually 3 - 5 lines of '%','%%' to port a Fastai `course v3` notebooks to run in DDP.

4. Extensible.  Porting to `fastai v2` is a high priority to-do.
5. Abstracted away the detail plumbing to harmonize `fastai`, `PyTorch DDP` and `ipyparallel` work with each other in the notebook environment.


## Installation:

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Overview and Examples:
### `Ddip` provides these iPython line and cell magics:
* `%load_ext Ddip`,  to load the extension
* `%makedip ...` to start/stop/restart a Distributed Data Parallel process group.  
* `%%dip ...` , to execute a cell in DDP, local notebook, or both.
* `%autodip ...`, to automatically execute subsequent cells in the DDP group, without requiring `%%dip` every time.
* `%dipush ...`, and `%dipull`, to pass things between the notebook and the DDP namespaces


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



