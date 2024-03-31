import os
import random

import torch
import numpy as np

from .config import TaskConfig, DistributedConfig
from utils import logging
from utils.logging import logger

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
    

    distributed = DistributedConfig(world_size, rank)
    return distributed


def set_logger(output_dir):
    # global logger

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # logger = logging.get_logger(os.path.join(output_dir, "log.txt"))
    logging.set_logger(os.path.join(output_dir, "log.txt"))
     

def retrieval_task(config: TaskConfig):
    local_rank = config.local_rank
    distributed = set_torch_cuda(config.seed, local_rank) 
    
   
    if local_rank == 0:
        parameters = config.dict().update(distributed.dict())
        logger.info("Effective parameters:")
        for key in sorted(parameters):
            logger.info("  <<< {}: {}".format(key, parameters.__dict__[key])) 
    
    
    