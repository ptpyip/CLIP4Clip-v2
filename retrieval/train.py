import math
import random
from typing import Tuple

import torch
import numpy as np

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
# from torch.nn.utils.clip_grad import clip_grad_norm_

from clip.clip import _transform as transform

from .config import *
from utils import logging
from utils.logging import logger

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def lr_cosine_with_warmup(t, warmup=0.002):
    if t < warmup:
        return t/warmup
    return 0.5 * (1.0 + math.cos(math.pi * t))

def clip_grad(model: nn.Module, clip_norm):
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip_norm)

def init_dataloaders() -> Tuple[DataLoader, int, DistributedSampler]:
    raise NotImplementedError
    return
    
def init_optimizer(
    model, lr, coef_lr, num_optimization_steps, warmup=-1, weight_decay = 0.2
):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        params=optimizer_grouped_parameters, lr=lr,
        betas=(0.9, 0.98), eps=1e-6,
        weight_decay=weight_decay,
        # max_grad_norm=1.0
    )
    scheduler = LambdaLR(optimizer,
        lr_lambda=lambda cur_iter: lr_cosine_with_warmup(
            cur_iter/num_optimization_steps, warmup
        )
    )

    return optimizer, scheduler
    

def train(
    model, config: TrainConfig, data_config: DataConfig, local_rank, 
    resume_ckpt_path=None
):
    sampler: DistributedSampler
    dataloader, sample_size, sampler = init_dataloaders(data_config)
    
    assert len(dataloader) == config.batch_size 
    num_optimization_steps = (
        int(config.batch_size + config.gradient_accumulation_steps - 1)
         / config.gradient_accumulation_steps
    ) * config.epochs
    optimizer, scheduler = init_optimizer(
        model, config, config.lr, config.coef_lr, 
        num_optimization_steps, config.warmup_proportion
    ) 
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, 
        find_unused_parameters=True
    )
    
    if local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", sample_size)
        logger.info("  Batch size = %d", config.batch_size)
        logger.info("  Num steps = %d", num_optimization_steps * config.gradient_accumulation_steps)

    start_epoch = 0
    best_score = 0.00001
    best_output_model_file = "None"
    
    ### resume optimizer state besides loss to continue train
    if resume_ckpt_path is not None:
        checkpoint = torch.load(resume_ckpt_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        # loss = checkpoint['loss']
        
    global_step = 0
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        
        loss, global_step = train_epoch()
        if local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, config.epochs, loss)

            # output_model_file = save_model(epoch, config, model, optimizer, loss, type_name="")

            ## Run on val dataset, this process is *TIME-consuming*.
            # logger.info("Eval on val dataset")
            # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

            # R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
            # if best_score <= R1:
            #     best_score = R1
            #     best_output_model_file = output_model_file
            # logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
        
    # clip_grad_norm_()
    # step
    
def train_epoch():
    ...    