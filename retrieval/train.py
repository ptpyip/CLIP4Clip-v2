import math
import time
import os.path

import torch
import numpy as np

from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clip.clip import _transform as transform

from .config import *
from .utils import logging
from .utils.logging import logger
# from .eval import EvalEpochCallable
from .dataloader import  init_dataloader

from model import CLIP4Clip

### Helpers
def lr_cosine_with_warmup(t, warmup=0.002):
    if t < warmup:
        return t/warmup
    return 0.5 * (1.0 + math.cos(math.pi * t))

def clip_grad(model: nn.Module, clip_norm):
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip_norm)


def train_epoch(
    model: CLIP4Clip, dataloader: DataLoader, 
    optimizer: Optimizer, scheduler: LambdaLR, 
    config: TrainConfig, global_step, train_log,  
    device, local_rank, n_gpu=0
):
    torch.cuda.empty_cache() 
    model.train()
   
    
    total_loss = 0.
    start_time = time.time()
    log_step = config.n_display
    gradient_accumulation_steps = config.gradient_accumulation_steps 
    
    for step, batch in enumerate(dataloader, start=1):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = (t.to(device=device, non_blocking=True) for t in batch)
        
        text, video, video_mask = batch
        loss: torch.Tensor = model(text, video, video_mask)
        
        if n_gpu > 1:
            loss = loss.mean()                          # average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward()
        total_loss += float(loss.item())
        
        ## skip if not enough steps
        if step % gradient_accumulation_steps == 0:
            ## update optimizer and lr
            clip_grad(model, 1.0)
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            
            # https://github.com/openai/CLIP/issues/46
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
                
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                lr = f"{scheduler.get_lr():.9f}"
                time_per_epoch = (time.time() - start_time) / (log_step * gradient_accumulation_steps)
                train_log(step, lr, loss, time_per_epoch)
                
                start_time = time.time()
   
    total_loss = total_loss / len(dataloader)
    return total_loss, global_step 
    

def train(
    model, config: TrainConfig, data_config: DataConfig,
    tokenizer, eval_epoch, device, local_rank, n_gpu=0,
    resume_ckpt_path=None, save_dir=None
):
    # sampler: DistributedSampler
    dataloader, sample_size, sampler = init_dataloader(
        data_config, "train", tokenizer, n_gpu, distributed=True
    )
    assert isinstance(sampler, DistributedSampler)
    
    # assert len(dataloader) == config.batch_size 
    num_optimization_steps = (
        int(config.batch_size + config.gradient_accumulation_steps - 1)
         / config.gradient_accumulation_steps
    ) * config.epochs
    optimizer, scheduler = init_optimizer(
        model, config.lr, config.coef_lr, 
        num_optimization_steps, config.warmup_proportion
    ) 
    
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    ) 
    
    if local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", sample_size)
        logger.info("  Batch size = %d", config.batch_size)
        logger.info("  Num steps = %d", num_optimization_steps * config.gradient_accumulation_steps)

    start_epoch = 0
    best_score = 1e-9
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
        
        train_log = lambda step, lr, loss, time_per_epoch : logger.info(
            ", ".join((
               f"Epoch: {epoch}/{config.epochs}",
               f"Step: {step}/{len(dataloader)}",
               f"Lr: {lr}", f"Loss: {loss}", f"Time/step: {time_per_epoch}" 
            ))
            # f"Epoch: {epoch}/{config.epochs}, Step: {step}/{config.batch_size}, Lr: {lr}, Loss: {loss}, Time/step: {time}"
        )
        
        loss, global_step = train_epoch(
            model, dataloader, optimizer, scheduler, config,                    # type: ignore
            global_step, train_log, device, local_rank, n_gpu
        )
        
        if local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, config.epochs, loss)
        
            if save_dir is not None:
                output_model_file = save_model(model, save_dir, epoch, optimizer, loss)

            
            tv_metrics, _ = eval_epoch(model, device, n_gpu, video_to_text_eval=False)                               # type: ignore
            R1 = tv_metrics['R1']
            if best_score <= R1:
                best_score = R1
                best_output_model_file = output_model_file
            logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            
    
def init_optimizer(
    model, lr, coef_lr, num_optimization_steps, warmup: float =-1, weight_decay = 0.2
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


def save_model(model, output_dir, epoch, optimizer, tr_loss):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    
    output_model_file = os.path.join(
        output_dir, f"{model.name}.bin.{epoch}"
    )
    optimizer_state_file = os.path.join(
        output_dir, f"{model.name}_opt.bin.{epoch}"
    )
    
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
    }, optimizer_state_file)
    
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file
