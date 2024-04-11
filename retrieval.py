"""This file contains the code for training and eval on retrieval task"""

import os
import yaml
import argparse

import torch
import torch.distributed

from clip.clip import _transform as transform

from retrieval.config import *
from retrieval import retrieval_task

from model import CLIP4Clip

MODELS = [
    "meanP-ViT-B/16","meanP-ViT-B/32",
    # "maxP-ViT-B/16","maxP-ViT-B/32",
    # "Trans-ViT-B/16","Trans-ViT-B/32"
]

torch.distributed.init_process_group(backend="nccl")


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")        ## set by torch.distributed.launch 
    parser.add_argument("--config", type=str, default='config/meanP-ViT-B-16-0326.yaml', help='')
    parser.add_argument("--data-dir", type=str, default='./data/msrvtt_data', help='')
    parser.add_argument("--video-dir", type=str, default='/csproject/dan3/data/msrvtt/videos', help='')
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()
    
    assert os.path.exists(args.config)
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    ## parse config   
    # clip_config = CLIPConfig,p
    
    model_config = ModelConfig(
        name = config_dict["model"].get("name"),
        temporal_mode = config_dict["model"]["temporal"].get("mode")
    )
    
    train_config = None
    is_train = config_dict.get("train") is not None
    if is_train:
        train_config = TrainConfig.parse_obj(config_dict["train"])
        
    dataset_config = DataConfig.parse_obj(config_dict["data"] | {
        "data_dir": args.data_dir,
        "video_dir": args.video_dir
    })
    
    task_config = TaskConfig(
        model=model_config, train=train_config, data=dataset_config, 
        local_rank=args.local_rank, seed=config_dict.get("seed", 0)
    )
    # print(dataset_config)    
    
    retrieval_task(task_config, is_train, distributed=torch.distributed.is_initialized())
    
     
if __name__ == "__main__":
    main()
  