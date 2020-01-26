# On Interactive, Distributed Training in FastAI Jupyter Notebooks

`ddipp` ("D dip") is a little iPython extension of line and cell magics to bring together `fastai` lesson notebooks [[1]](#course_v3) and PyTorch's Distributed Data Parallel [[2]](#pytorchddp).  It uses `ipyparallel` [[3]](#ipp) to manage the DDP process group. 

Platform tested: single host with multiple GPU, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.


## Installation:

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Quick Usage Examples in Jupyter Notebook:

1. Load the extension: `%load_ext ddipp`

2. Initialize GPUs, fastai, and DDP:  `%makedip -g all -a fastai_v1 --verbose True`

Now the Jupyter notebook is ready to dance between the local iPython and the DDP processes:

* Line magic `%dip --push a b c` to push local notebook variables `a`,`b`, and `c` into the DDP processes.

* Cell magic `%%dip` to execute a notebook cell in parallel on the DDP processes.  Use option `--to both` to run the cell in both locally *and* on DDP.

* Line magic `%autodip on` to implicitly prepend `%%dip` to all subsequent cells, thus automatically run them all on the DDP processes. 

* Insert `#dip_locallly` as the cell's first line, to enforce local execution, regardless `%autodip` is on or off.

## Documentation:
* [`ddipp` Manual](Manual.md)
* [Example Notebooks](nbs/)
* [Known Limitations, Issues, Bugs and Features to Add](Issues.md)

## References:

1. <a name="course_v3"></a> [FastAI Course v3](https://course.fast.ai/)

2. <a name="pytorchddp"></a>On Distributed training:
* [Tutorial from PyTorch on Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), 
* [Launching fastai to use DDP](https://docs.fast.ai/distributed.html), *FastAI*
* Further readings: [PyTorch Lightning -- Tips for faster training ](https://towardsdatascience.com/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565), [On the performance of different training parallelism](http://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)


3. <a name="ipp"></a>On `ipyparallel`, a few resources and inspirations to `ddipp`:
* [The official documentation](https://ipyparallel.readthedocs.io/en/latest/intro.html),
* [An intro to ipyparallel](http://activisiongamescience.github.io/2016/04/19/IPython-Parallel-Introduction/), *Activevision Game Science*
* ["Using ipyparallel"](http://people.duke.edu/~ccc14/sta-663-2016/19C_IPyParallel.html), *Duke University, "Computational Statistics in Python"*
* [Interactive Distributed Deep Learning with Jupyter Notebooks](https://sc18.supercomputing.org/proceedings/tech_poster/poster_files/post206s2-file3.pdf), *Lawrence Berkeley National Laboratory/Cray Inc.*



