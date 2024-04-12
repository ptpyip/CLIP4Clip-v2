import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from typing import Tuple, Optional
from ..config import DataConfig
from .retrievalDataset import RetrievalDataset
from .msrvtt import MSRVTTDataset

from clip import tokenize

DATASETS = {
   "msrvtt" : MSRVTTDataset
}

class RetrievalDataLoader(DataLoader):
    dataset: RetrievalDataset
    
    
def init_dataloader(
    config: DataConfig, subset, tokenizer, n_gpu, distributed=False
) -> Tuple[RetrievalDataLoader, int, DistributedSampler | None]:
    Dataset = DATASETS[config.dataset]
    is_train = (subset == "train")
    batch_size = config.train_batch_size if is_train else config.eval_batch_size
    
    # if batch_size % n_gpu != 0: raise ValueError(
    #     f"Invalid batch_size_{subset} and n_gpu parameter: {batch_size}%{n_gpu} should be == 0"
    # )
    
    dataset = Dataset(
        subset, config.data_dir, config.video_dir, tokenizer, 
        max_words=config.max_words, max_frames=config.max_frames,
    )

    sampler = DistributedSampler(dataset) if (is_train and distributed) else None
    
    dataloader = RetrievalDataLoader(
        dataset,
        batch_size=batch_size,      # batch_size per gpu 
        num_workers=config.num_thread_reader,
        shuffle=(is_train and not distributed),
        sampler=sampler,
        drop_last=is_train,
    )

    return dataloader, len(dataset), sampler