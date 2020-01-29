# Interactive PyTorch DDP Training in FastAI Jupyter Notebooks

`Ddip` ("Dee dip") --- Distributed Data "interactive" Parallel is a little iPython extension of line and cell magics to bring together `fastai` lesson notebooks [[1]](#course_v3) and PyTorch's Distributed Data Parallel [[2]](#pytorchddp).  It uses `ipyparallel` [[3]](#ipp) to manage the DDP process group. 

Platform tested: single host with multiple Nvidia CUDA GPUs, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.

## Features:

`Ddip` was designed to make experiments with multiple GPU/Distributed Data Parallel training a little bit easier in Fastai notebooks. Towards that end:

1. Parallel execution in a multiprocess PyTorch DDP group, separate from the local notebook namespace
2. Automatic garbage collection and freeing GPU cache memory after parallel execution of a cell
3. Minimize the code changes required Fastai notebooks -- often only 3 - 5 new lines of code.
4. Stream any parallel execution output in real time to notebook console.


## Installation:

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Overview and Examples:
### `Ddip` line and cell magics
* `%load_ext Ddip`,  to load the extension
* `%makedip -g all -a fastai_v1`, to initialize `fastai` to use DDP in notebook
* `%%dip {remote | local | everywhere} `, to designate where a cell is executed
* `%dipush bs src .....`, to push some variables from notebook to the DDP processes
* `%autodip {on | off}`, to switch on/off automatic remote-execution of subsequent cells

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



