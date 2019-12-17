#!/usr/bin/env python3
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from fastai.callbacks.mem import PeakMemMetric
import argparse, importlib, torch

def petsmore_res34():
    bs = 64
    path = untar_data(URLs.PETS)/'images'
    tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)
    src = ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)
    def get_data(size, bs, padding_mode='reflection'):
        return (src.label_from_re(r'([^/]+)_\d+.jpg$')
            .transform(tfms, size=size, padding_mode=padding_mode)
            .databunch(bs=bs).normalize(imagenet_stats))
    data = get_data(224,bs)
    gc.collect()
    learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, callback_fns=PeakMemMetric)
    # learn = learn.to_distributed(rank_distrib())
    learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)

def cifar_wrn22():
    path = untar_data(URLs.CIFAR)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=256).normalize(cifar_stats)
    learn = Learner(data, wrn_22(), metrics=accuracy) # .to_distributed(rank_distrib())
    learn.fit_one_cycle(2, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

def cifar_wrn22_dist():
    path = untar_data(URLs.CIFAR)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=256).normalize(cifar_stats)
    learn = Learner(data, wrn_22(), metrics=accuracy).to_distributed(rank_distrib())
    learn.fit_one_cycle(2, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--code", type=str, help="pyscript which has a train() function defined.")
    # parser.add_argument("--nvprof")
    args = parser.parse_args()
    r = args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    for t in args.code.split(','):
        t_func = globals().get(t, None)
        if t_func:
            print(f"Training Rank [{r}]: {t}", flush=True)
            learner = t_func()
            del learner
            distrib_barrier()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Training function {t} undefined")
    
    torch.distributed.destroy_process_group()
    