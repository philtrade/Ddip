# ipyparallel-torchddp
IPython magics to use ipyparallel (IPP) to harness PyTorch distributed data parallel training (DDP).
The FastAI course-V3 is the first use case.

Installations

`pip install git+https://github.com/philtrade/ipyparallel-torchddp.git`

Usage in a Jupyter Notebook

1. Load the extension:
```
%load_ext ddipp

%makedip -g all -a fastai_v1 --verbose True
```

Output:

```
Waiting for connection file: ~/.ipython/profile_default/security/ipcontroller-ippdpp_c-client.json
Connecting to ipyparallel cluster.......
Initializing torch distributed group with GPUs [0, 1, 2]
Local Ranks initialized:  ['GPU0=0', 'GPU1=1', 'GPU2=2']
Importing on cluster: import fastai, fastai.torch_core, torch, fastprogress
from fastai.distributed import *
from ddipp.fastai_v1 import initializer, finalizer, set_verbose, lr_find_bypass
fastai_v1:
[Process 30097] Rank 0 fastai initialized for distributed data parallel.
[Process 30101] Rank 1 fastai initialized for distributed data parallel.
[Process 30103] Rank 2 fastai initialized for distributed data parallel.
```
2.