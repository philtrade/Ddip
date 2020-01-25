# On Using Distributed Data Parallel in FastAI Jupyter Notebooks

`ddipp` ("D dip") is a little iPython extension of line and cell magics.  It uses `ipyparallel` to manage the process pool of DDP inside iPython/Jupyter notebook environment, and patches up `fastai v1` at runtime, so that the notebooks in FastAI's course-v3 can train models using multiple GPUs, interactively (which it can't as of late 2019).


Platform tested: single host with multiple GPU, Ubuntu linux + PyTorch + Python 3, fastai v1 and fastai course-v3.


## Installation:

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Quick Usage Examples:

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

- [FastAI Course v3](https://course.fast.ai/)
- [About PyTorch's DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), and [Launching fastai to use DDP](https://docs.fast.ai/distributed.html)
- [A gentle intro to using ipyparallel](http://people.duke.edu/~ccc14/sta-663-2016/19C_IPyParallel.html) and [the official ipyparallel documentation](https://ipyparallel.readthedocs.io/en/latest/intro.html)



     