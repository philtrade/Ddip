#### Rough idea of a multiprocess spawning context manager to facilitate on-demand DDP in jupyter notebook via function calls instead of relying on ipython cell magics.

References:
* General structure follows https://pytorch.org/tutorials/intermediate/dist_tuto.html and https://pytorch.org/docs/stable/notes/multiprocessing.html
* On using `multiprocess` instead of `multiprocessing` and `torch.multiprocessing`: https://hpc-carpentry.github.io/hpc-python/06-parallel/ 
* On `from module import *` within a function: https://stackoverflow.com/questions/41990900/what-is-the-function-form-of-star-import-in-python-3


```code
from multiprocess import Process  # use multiprocess, not python's multiprocessing, nor torch.multiprocessing.

builder = lambda : learner_builder(path, ….) # state dict can be passed in here….
trainer = lambda learner: learner.fit(……)

# generalize *_fn into a pipeline of funcs/tasks or steps.

def init_process(rank, imports, builder_fn, trainer_fn):
	""" Initialize the child process environment. """
    def starred_imports(modules):
        for mod in modules:
            m = __import__(mod, fromlist=[‘*’])
            
            if hasattr(m, '__all__'): all_names = m.__all__
            else: all_names = [n for n in dir(m) if not n.startswith('_')]

            globals().update({n: getattr(m, n) for n in all_names})

	import os
	starred_imports(imports + [‘fastai.distributed’]) # can rid of fastai.* if can reimplement distrib_ctx() here.

  """ Initialize the distributed environment. """
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  os.environ[‘WORLD_SIZE’] = str(self.ws)
  ….
  <more os.environ setup, without initialize_group()>

	learner = builder_fn()

	# distrib_ctx() will setup/destroy ddp group upon enter/exit.

    with distrib_ctx(learner):
    		trainer_fn(learner)

	<cleanup os.environ[….] setup >
    return learner


 class DDPTrainCtx():
   def __init__(self, ws, imports, builder_fn, train_fn):
	self.ws = ws
	self.procs = []
        self.imports = imports # [ ‘list’, ‘of’, ‘module.names’ ]
        self.builder = builder_fn
        self.trainer = trainer_fn

  def enter():
    for rank in range(self.ws):
        p = Process(target=init_process,
		    args=(rank, self.imports, self.builder, self.trainer))
        self.procs.append(p)

  def run():
	# Start spawned procs in background first…
	for p in self.procs: p.start()

	# Then in foreground, in the parent rank-0 process
	init_process(0, self.imports, self.builder, self.trainer)

  def exit():
    for p in processes: p.join()
```

