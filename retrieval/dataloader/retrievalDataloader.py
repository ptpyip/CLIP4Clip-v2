import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from typing import Tuple
from .retrievalDataset import RetrievalDataset

from ..config import DataConfig

class RetrievalDataLoader(DataLoader):
    dataset: RetrievalDataset



def init_dataloader(data_config, mode) -> Tuple[RetrievalDataLoader, int, DistributedSampler]:
    raise NotImplementedError