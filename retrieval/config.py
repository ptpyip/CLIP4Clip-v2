from pydantic import BaseSettings
from typing import Optional
from enum import Enum

class Dataset(Enum):
    MSVD = "msvd"
    MRVTT = "msrvtt"
    LSMDC = "lsmdc"
    

class CLIPConfig(BaseSettings):
    name: str
    pretrained: bool = True
    freeze_layer_num: int
    

class ModelConfig(BaseSettings):
    name: str
    clip: CLIPConfig
    ckpt_path: Optional[str] = None
    

class TrainConfig(BaseSettings):
    epochs: int
    lr: float
    coef_lr: float =1.
    batch_size: int
    n_display: int
    warmup_proportion: float = 0.1 
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    gradient_accumulation_steps: int = 1    
    # reduce batch size by k, to fit data into VRAM. Run K times before update
   
    
class DistributedConfig(BaseSettings):
    world_size: int
    rank: int

class DataConfig(BaseSettings):
    dataset: Dataset
    data_path: str
    video_path: str
    frame_rate: int = 1
    max_words: int = 32
    max_frames: int = 10
    num_thread_reader: int = 1
    train_batch_size: Optional[int]
    eval_batch_size: Optional[int]

class TaskConfig(BaseSettings):
    seed: int
    local_rank: int             # specified in args (controlled by torch.distributed.launch )
    model: ModelConfig
    train: TrainConfig

    