import os
import random

import torch
import torch.backends.cudnn
import torch.distributed
import numpy as np

from clip.clip import _transform as transform
from clip import tokenize

from .config import *
from .utils import logging
from .utils.logging import logger

from model import build_model, CLIPTokenizer
from .train import train
from .eval import get_eval_epoch

def set_torch_cuda(seed, local_rank):
    # global logger
    # predefining random initial seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)        # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
    torch.cuda.set_device(local_rank)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    distributed = DistributedConfig(world_size=world_size, rank=rank)
    return distributed


def set_logger(output_dir):
    # global logger

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # logger = logging.get_logger(os.path.join(output_dir, "log.txt"))
    logging.set_logger(os.path.join(output_dir, "log.txt"))
     

def init_device(local_rank):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", local_rank
    )

    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device} n_gpu: {n_gpu}")

    return device, n_gpu

def init_model(
    config: ModelConfig, device: torch.device | str ="cpu", is_training=False,
    world_size=None, rank=None
):
    """init a CLIP4Clip model"""
    state_dict = {}
    if config.ckpt_path != None:
        if not os.path.exists(config.ckpt_path):
            raise RuntimeError(
                f"Model {config.name} not found with path: {config.ckpt_path}"
            )
        state_dict = torch.load(config.ckpt_path, map_location=device)
        
    model = build_model(config, state_dict, world_size, rank)
    
    ### freeze testing
    # freeze_layer_num = config.clip.freeze_layer_num 
    freeze_layer_num = 0
    assert freeze_layer_num <= 12 and freeze_layer_num >= -1
    if is_training and freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if (name.find("ln_final.") == 0 or name.find("text_projection") == 0 
                or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0
                or name.find("logit_scale") == 0
            ):
                continue    # need to train
            
            elif (name.find("visual.transformer.resblocks.") == 0 
                or name.find("transformer.resblocks.") == 0
            ):
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= freeze_layer_num:
                    continue    # need to train
                
            param.requires_grad = False
    
    model.to(device)    
    return model, transform(model.input_resolution)


def retrieval_task(config: TaskConfig, is_train=True, distributed=True):
    assert (config.train is not None) == is_train
    local_rank = config.local_rank
    distribute_config = set_torch_cuda(config.seed, local_rank) 
    set_logger(output_dir="logs")
    
    if local_rank == 0:
        parameters = config.dict() | distribute_config.dict()
        logger.info("Effective parameters:")
        for key in sorted(parameters):
            logger.info("  <<< {}: {}".format(key, parameters[key])) 
    
    device, n_gpu = init_device(local_rank)
    model, _ = init_model(config.model, device, is_training=True, **distribute_config.dict())
    """ transform is done on video extraction."""
    
    tokenizer = CLIPTokenizer()
    
    # dataloaders = init_dataloaders()

    eval_epoch = get_eval_epoch(
        config.data, tokenizer, local_rank, n_gpu, distributed
    )
    if is_train:
        assert config.train is not None
        train(
            model, config.train, config.data, tokenizer, 
            eval_epoch, device, local_rank, n_gpu, 
            save_dir="./ckpts"
        )
    else:
        if local_rank !=0:
            pass
        eval_epoch(model=model, device=device, n_gpu=n_gpu)
        
        
    
    
    
     
    