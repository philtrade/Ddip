# ipyparallel-torchddp
IPython magics to use ipyparallel (IPP) to harness PyTorch distributed data parallel training (DDP).
The FastAI course-V3 is the first use case.

## Installations

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

## Usage

### Load the extension:
```
%load_ext ddipp
```

### Commands/Magics in iPython/Jupyter Notebook

* Initialize a Distributed Data Parallel processing group with a list of GPUs, and prepare to run FastAi v1's library in the notebook, on this DDP group:

```
%makedip -g all -a fastai_v1 --verbose True
```
* Passing variables between notebook iPython and the DDP procecesses

```
%dip --push var1 var2 var3..... varN
```
* Execute a cell in the DDP group (remote):
```
%%dip # Add this to the first line of a notebook cell
<code>
```

* Execute a cell both in the notebook namespace (local) and in the DDP group (remote):
```
%%dip --to both
from fastai.vision import *
import os
```

* Turn on/off automatic execution in the DDP group:
```
%autodip [on or off] [optional flags passed to %%dip]

< all subsequent cells are executed in the remote DDP group only, as if every cell starts with %%dip on the first line>
```

* In the middle when `%autodip` is on, want to execute a cell locally in the notebook namespace:

```
%%dip --to local
<cell to be run only in local namespace, regardless %autodip is on or off>
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





     