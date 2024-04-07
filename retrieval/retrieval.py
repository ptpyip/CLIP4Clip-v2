import os
import random

import torch
import torch.backends.cudnn
import torch.distributed
import numpy as np

from clip.clip import _transform as transform

from .config import *
from utils import logging
from utils.logging import logger

from ..model import build_model, CLIPTokenizer

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
     

def init_device(config: TaskConfig, local_rank):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", local_rank
    )

    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device} n_gpu: {n_gpu}")
    # args.n_gpu = n_gpu

    if (config.train.batch_size % n_gpu != 0 
        or config.eval.batch_size % n_gpu != 0
    ):
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            config.train.batch_size, n_gpu, config.eval.batch_size, n_gpu)
        )

    return device, n_gpu

def init_model(
    config: ModelConfig, device: torch.device | str ="cpu", is_training=False
):
    """init a CLIP4Clip model"""
    state_dict = {}
    if config.ckpt_path != None:
        if not os.path.exists(config.ckpt_path):
            raise RuntimeError(
                f"Model {config.name} not found with path: {config.ckpt_path}"
            )
        state_dict = torch.load(config.ckpt_path, map_location=device)
        
    model = build_model(config.name, state_dict)
    
    ### freeze testing
    freeze_layer_num = config.clip.freeze_layer_num 
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
    
    return model, transform(model.input_resolution)

def init_dataloaders():
    ...


def retrieval_task(config: TaskConfig):
    local_rank = config.local_rank
    distributed = set_torch_cuda(config.seed, local_rank) 
    
    if local_rank == 0:
        parameters = config.dict() | distributed.dict()
        logger.info("Effective parameters:")
        for key in sorted(parameters):
            logger.info("  <<< {}: {}".format(key, parameters.__dict__[key])) 
    
    device, n_gpu = init_device(config, local_rank)
    model = init_model(config.model, device, is_training=True)
    
    tokenizer = CLIPTokenizer()
    # dataloaders = init_dataloaders()
    
    
    
     
    