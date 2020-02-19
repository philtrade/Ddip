
# Known Issues and Things to Improve

    "..finish something.... like properly, finish it."
> > > *Jeremey Howard*, [2:07:40, Lesson 7, Deep Learning Part 1 Course V3](https://www.youtube.com/watch?v=9spwoDYwW_I&feature=youtu.be&t=7660).

The journey to build `Ddip` has been incredibly fruitful, and yet plenty more fruits to pick ahead.  Among them, a few highlights:

> * Build a module to work with `fastai v2`, the new version of fastai as of year 2020.
> * Try the **`nbdev`** development approach.
> * ... 

*and I am sure you and the community have many great ideas, so I invite you to send me feedbacks, thoughts, questions, and constructive criticisms.*


## Limitations and Bugs

* **Works in Single Host with Multiple GPU only**

    `Ddip` was developed/tested on a single host with multiple GPUs, and may not work in "*multiple nodes x multiple GPU*" yet.  But the underlying fabric `ipyparallel` does support distributed, parallel computing well, so it should be doable.

* **Process group must starts with GPU 0, and in consecutive, ascending order**

    In other words, on a host with GPU [0,1,2,3], the following DDP group specification may not work properly yet: 
    `3,2,1,0`, or `1,2,3`, or `0,3` etc...

**Issues with FastAI notebooks:**

* Performance issues: lesson6-pets-more no speedup with DDP, lesson7-wgan no speedup with DDP.  lesson3-imdb slows down first with 2 GPUs, then speed up.

 - [X] Fixed in [PR #2498](https://github.com/fastai/fastai/pull/2498) lesson7-wgan shows linear slow-down (3X training time with 3 GPUs vs 1 GPU)

 - [X] [Issue #2501](https://github.com/fastai/fastai/issues/2501) lesson3-imdb, a langauge modelling task, doesn't seem to train correctly when in multiple-GPU DDP mode.

        When loaded with a previously trained encoder `fine_tuned_enc`, the next `learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))` accuracy is stuck at 49.xx%, not the supposed 92%+.

## Things to Add:
* **Unit tests, and perhaps CI**

    So far FastAi's lesson notebooks are used as integration test.  `ddipp` needs and will benefit from some unit tests.
* **Support for DDP in multiple nodes x multiple GPUs configuration**

- [X] Fixed in [PR #2487](https://github.com/fastai/fastai/pull/2487): Reimplement learning rate finder bypass
* Automatic garbage collection using fastai's callback mechanism?
    
    Current impementation monkey-patches up the code at runtime.  Using `fastai`'s callback maybe a much cleaner solution.  E.g. only do garbage collection at the end of a fit(), instead of only at the end of a notebook cell execution.

* **FastAi V2**
    
    `ddipp` was developed using FastAi v1's lesson notebooks, and `fastai v2` is being rolled out. `ddipp` may need to catch up.

* **FastAi's `nbdev` Approach**
    
    *[Use Jupyter Notebook for Everything](https://www.fast.ai/2019/12/02/nbdev/)* - Jeremy Howard, Dec 2, 2019

    Centuries of wisdom from experimental science distilled from the many great minds, all point to the benefit of iteractive tinkering and collaborative research via documentation and tests. 
    
    In software development of this age, finally a pragmatic tool to foster one-stop development of code, documentation, and tests is emerging --- `nbdev`, the labor of love from the brains behind FastAi.

    `ddipp` needs better documentation, more unit tests, and there are new features to explore -- for those future tasks, it would be interesting to see the productivity gain by trying out `nbdev`.


* **Support for Fastai in Swift**  *Gasp!*





