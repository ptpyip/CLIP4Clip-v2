from pydantic import BaseSettings

class ModelConfig(BaseSettings):
    name: str
    
    
class TrainConfig(BaseSettings):
    epochs: int
    lr: float
    coef_lr: float
    batch_size: int
    n_display: int
    
class DistributedConfig(BaseSettings):
    world_size: int
    rank: int


class TaskConfig(BaseSettings):
    seed: int
    local_rank: int             # specified in args (controlled by torch.distributed.launch )
    model: ModelConfig
    train: TrainConfig

    